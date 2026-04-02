# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import Namespace
from decimal import Decimal
import logging
from math import ceil
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple, Union
import warnings

import meeteval
from meeteval.io.seglst import SegLstSegment, SegLST
from meeteval.io.rttm import RTTM, RTTMLine
import torch
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


from nemo.collections.asr.parts.submodules.ctc_decoding import AbstractCTCDecoding
from nemo.collections.asr.parts.submodules.multitask_decoding import AbstractMultiTaskDecoding
from nemo.collections.asr.parts.submodules.rnnt_decoding import AbstractRNNTDecoding
from nemo.utils.get_rank import is_global_rank_zero, get_rank
from nemo.utils.distributed import get_world_size
from nemo.collections.asr.data.text_norm import get_text_norm

__all__ = ['MeetevalDER']

logging.getLogger('meeteval').setLevel(logging.INFO)

class MeetevalDER(Metric):
    full_state_update: bool = True

    def __init__(
        self,
        batch_dim_index=0,
        dist_sync_on_step=False,
        fold_consecutive=True,
        sync_on_compute=True,
        embed_duration=0.08, # 80ms - 12.5hz with 8x downsampling conformer
        threshold=0.5,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step, sync_on_compute=sync_on_compute)

        self.fold_consecutive = fold_consecutive
        self.batch_dim_index = batch_dim_index
        self.embed_duration = embed_duration
        self.threshold = threshold

        self.per_utt_data = dict()

    def _tensor_to_segments(self, tensor: torch.Tensor, utt_id, offset):
        # offset is in seconds
        segments = []
        
        if isinstance(offset, torch.Tensor):
            offset = offset.item()

        # Loop through speakers
        for i in range(tensor.shape[-1]):
            # 1 - start, -1 - end
            diff = (torch.concat([tensor[:, i].float(), torch.zeros((1, ), device=tensor.device)]) - torch.concat([torch.zeros((1, ), device=tensor.device), tensor[:, i].float()]))
            starts = torch.where(diff == 1)[0]
            ends = torch.where(diff == -1)[0]
            assert len(starts) == len(ends)
            for j in range(len(starts)):
                segments.append(RTTMLine(type='SPEAKER', filename=utt_id, channel=0, begin_time=offset + starts[j].item()*self.embed_duration, duration=(ends[j]-starts[j]).item()*self.embed_duration, orthography='<NA>', speaker_type='<NA>', speaker_id=i, confidence='<NA>', signal_look_ahead_time='<NA>'))
        return segments

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_lens: torch.Tensor,
        utt_ids: torch.Tensor,
        offsets: torch.Tensor,
        rttm_file_paths: List[str],
    ):
        
        assert len(predictions) == len(targets) == len(target_lens) == len(utt_ids) == len(offsets) == len(rttm_file_paths)

        # predictions: [B, T, S]
        with torch.no_grad():
            preds = predictions > self.threshold
            for pred, target, target_len, utt_id, offset, rttm_file_path in zip(preds, targets, target_lens, utt_ids, offsets, rttm_file_paths):
                pred = pred[:target_len]
                target = target[:target_len]
                present_pred_speakers = pred.sum(dim=0) != 0
                present_target_speakers = target.sum(dim=0) != 0
                pred = pred[:, present_pred_speakers]
                target = target[:, present_target_speakers]
                # for i in range(pred.shape[-1])

                utt_id = f'{utt_id}_{str(offset.item()).replace(".", "_")}'
                pred_segments = self._tensor_to_segments(pred, utt_id, 0)

                if utt_id not in self.per_utt_data:
                    self.per_utt_data[utt_id] = dict()
                    # self.per_utt_data[utt_id]['target_segments'] = meeteval.io.load(rttm_file_path)
                    self.per_utt_data[utt_id]['target_segments'] = self._tensor_to_segments(target, utt_id, 0)

                if 'pred_segments' not in self.per_utt_data[utt_id]:
                    self.per_utt_data[utt_id]['pred_segments'] = []
                self.per_utt_data[utt_id]['pred_segments'].extend(pred_segments)

    def _process_metric_res(self, per_item_res: List[Dict]):
        res = {'scored_speaker_time': 0, 'missed_speaker_time': 0, 'falarm_speaker_time': 0, 'speaker_error_time': 0}
        for _, item_res in per_item_res.items():
            # Meeteval returns a dict with utt_id as key and the error as value.
            # We're scoring per-item so it's always a single item dict.
            item_res = list(item_res.values())[0]
            res['scored_speaker_time'] += item_res.scored_speaker_time
            res['missed_speaker_time'] += item_res.missed_speaker_time
            res['falarm_speaker_time'] += item_res.falarm_speaker_time
            res['speaker_error_time'] += item_res.speaker_error_time
        return res
    
    # def _reduce_res(self, res_all_ranks: List[Dict]):
    #     res = {'scored_speaker_time': 0, 'missed_speaker_time': 0, 'falarm_speaker_time': 0, 'speaker_error_time': 0}
    #     for res_rank in res_all_ranks:
    #         for k in res:
    #             res[k] += res_rank[k]
    #     return res

    def reduce_utterances(self, res_all_ranks):
        res = dict()
        for rank_res in res_all_ranks:
            for utt_id, res_der in rank_res:
                res[utt_id] = res_der
        return res

    def _reduce_res(self, res_all_ranks: List[Dict]):
        res = {'scored_speaker_time': 0, 'missed_speaker_time': 0, 'falarm_speaker_time': 0, 'speaker_error_time': 0}
        for res_rank in res_all_ranks:
            for k in res:
                res[k] += res_rank[k]
        return res
    
    def _write_pred_RTTM(self, rttm: RTTM, rttm_file_path: str):
        with open(rttm_file_path, 'w') as f:
            for line in rttm.lines:
                f.write(f'{line.type} {line.filename.split("_")[0]} {line.channel} {line.begin_time:.3f} {line.duration:.3f} {line.orthography} {line.speaker_type} {line.speaker_id} {line.confidence} {line.signal_look_ahead_time}\n')

    @staticmethod
    def _merge_rttm_segments(rttm_lines: List[RTTMLine]):
        rttm_lines.sort(key=lambda x: x.begin_time)
        per_spk_per_utt_rttm_lines = dict()
        res = []

        for line in rttm_lines:
            spk_utt_key = f"{line.speaker_id}_{line.filename}"
            if spk_utt_key not in per_spk_per_utt_rttm_lines:
                per_spk_per_utt_rttm_lines[spk_utt_key] = []
            per_spk_per_utt_rttm_lines[spk_utt_key].append(line)

        for spk_utt_key in per_spk_per_utt_rttm_lines:
            merged_rttm_lines = []
            current_start_time = per_spk_per_utt_rttm_lines[spk_utt_key][0].begin_time
            current_end_time = per_spk_per_utt_rttm_lines[spk_utt_key][0].begin_time + per_spk_per_utt_rttm_lines[spk_utt_key][0].duration

            for line in per_spk_per_utt_rttm_lines[spk_utt_key][1:]:
                if line.begin_time <= current_end_time:
                    current_end_time = max(current_end_time, line.begin_time + line.duration)
                else:
                    merged_rttm_lines.append(RTTMLine(type='SPEAKER', filename=line.filename, channel=line.channel, begin_time=current_start_time, duration=current_end_time - current_start_time, orthography=line.orthography, speaker_type=line.speaker_type, speaker_id=line.speaker_id, confidence=line.confidence, signal_look_ahead_time=line.signal_look_ahead_time))
                    current_start_time = line.begin_time
                    current_end_time = line.begin_time + line.duration

            merged_rttm_lines.append(RTTMLine(type='SPEAKER', filename=line.filename, channel=line.channel, begin_time=current_start_time, duration=current_end_time - current_start_time, orthography=line.orthography, speaker_type=line.speaker_type, speaker_id=line.speaker_id, confidence=line.confidence, signal_look_ahead_time=line.signal_look_ahead_time))
            res.extend(merged_rttm_lines)

        return res


    def compute(self, collar=0.0, pred_rttm_path=None):
        results = []
        for utt_id in self.per_utt_data:
            all_targets_rttm = RTTM(lines=self.per_utt_data[utt_id]['target_segments'])
            all_preds_rttm = RTTM(lines=self.per_utt_data[utt_id]['pred_segments'])
            if not all_preds_rttm.lines:
                # Not 100% correct. Some speakers may overlap with themselves (TODO: SOLVE).
                spk_time = sum(l.duration for l in all_targets_rttm.lines)
                res_der = {utt_id: Namespace(scored_speaker_time=Decimal(spk_time), missed_speaker_time=Decimal(spk_time), falarm_speaker_time=Decimal(0), speaker_error_time=Decimal(0))}
            else:
                try:
                    with warnings.catch_warnings(action="ignore"):
                        res_der = meeteval.der.dscore(reference=all_targets_rttm, hypothesis=all_preds_rttm, collar=collar)
                except Exception as e:
                    print(f"Error scoring {utt_id}: {e}")
                    spk_time = sum(l.duration for l in all_targets_rttm.lines)
                    res_der = {utt_id: Namespace(scored_speaker_time=Decimal(spk_time), missed_speaker_time=Decimal(spk_time), falarm_speaker_time=Decimal(0), speaker_error_time=Decimal(0))}
            results.append((utt_id, res_der))

        flattened_preds = []
        flattened_targets = []
        for uid in self.per_utt_data:
            flattened_preds.extend(self.per_utt_data[uid]['pred_segments'])
            flattened_targets.extend(self.per_utt_data[uid]['target_segments'])
        merged_preds = self._merge_rttm_segments(flattened_preds)
        merged_targets = self._merge_rttm_segments(flattened_targets)
        merged_pred_rttm = RTTM(lines=merged_preds)
        merged_target_rttm = RTTM(lines=merged_targets)

        # results = self._process_metric_res(results)

        res_all_ranks = [None] * get_world_size()
        if get_world_size() > 1:
            torch.distributed.all_gather_object(res_all_ranks, results)
        else:
            res_all_ranks[0] = results

        reduced_utts = self.reduce_utterances(res_all_ranks)
        res_all_ranks = self._process_metric_res(reduced_utts)
        res_all_ranks['der'] = (res_all_ranks['speaker_error_time'] + res_all_ranks['missed_speaker_time'] + res_all_ranks['falarm_speaker_time']) / res_all_ranks['scored_speaker_time']

        if pred_rttm_path is not None:
            self._write_pred_RTTM(merged_pred_rttm, rttm_file_path=pred_rttm_path)

        return res_all_ranks

    def reset(self):
        super().reset()
        self.per_utt_data.clear()
