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

from math import ceil
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple, Union

import meeteval
from meeteval.io.seglst import SegLstSegment, SegLST
import torch
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


from nemo.collections.asr.parts.submodules.ctc_decoding import AbstractCTCDecoding
from nemo.collections.asr.parts.submodules.multitask_decoding import AbstractMultiTaskDecoding
from nemo.collections.asr.parts.submodules.rnnt_decoding import AbstractRNNTDecoding
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero, get_rank
from nemo.utils.distributed import get_world_size

__all__ = ['MeetevalMTWER']


class MeetevalMTWER(Metric):
    full_state_update: bool = True

    def __init__(
        self,
        decoding: Union[AbstractCTCDecoding, AbstractRNNTDecoding, AbstractMultiTaskDecoding],
        use_cer=False,
        log_prediction=True,
        batch_dim_index=0,
        dist_sync_on_step=False,
        fold_consecutive=True,
        sync_on_compute=True,
        embed_duration=0.08, # 80ms - 12.5hz with 8x downsampling conformer
        output_per_word_timestamps=True,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step, sync_on_compute=sync_on_compute)

        self.decoding = decoding
        self.use_cer = use_cer
        self.log_prediction = log_prediction
        self.fold_consecutive = fold_consecutive
        self.batch_dim_index = batch_dim_index
        self.embed_duration = embed_duration
        self.output_per_word_timestamps = output_per_word_timestamps

        self.decode = None
        if isinstance(self.decoding, AbstractRNNTDecoding):
            self.decode = lambda predictions, predictions_lengths, predictions_mask, input_ids, targets: self.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=predictions, encoded_lengths=predictions_lengths,
                return_hypotheses=True,
            )
        elif isinstance(self.decoding, AbstractCTCDecoding):
            self.decode = lambda predictions, predictions_lengths, predictions_mask, input_ids, targets: self.decoding.ctc_decoder_predictions_tensor(
                decoder_outputs=predictions,
                decoder_lengths=predictions_lengths,
                fold_consecutive=self.fold_consecutive,
                return_hypotheses=True,
            )
        elif isinstance(self.decoding, AbstractMultiTaskDecoding):
            self.decode = lambda predictions, prediction_lengths, predictions_mask, input_ids, targets: self.decoding.decode_predictions_tensor(
                encoder_hidden_states=predictions,
                encoder_input_mask=predictions_mask,
                decoder_input_ids=input_ids,
                return_hypotheses=False,
            )
        else:
            raise TypeError(f"WER metric does not support decoding of type {type(self.decoding)}")

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("preds_lengths", default=[], dist_reduce_fx="cat")
        self.add_state("preds_word_timestamps", default=[], dist_reduce_fx="cat")
        self.add_state("preds_word_timestamps_lengths", default=[], dist_reduce_fx="cat")
        self.add_state("utt_ids", default=[], dist_reduce_fx="cat")
        self.add_state("spk_ids", default=[], dist_reduce_fx="cat")

    def update(
        self,
        predictions: torch.Tensor,
        predictions_lengths: torch.Tensor,
        utt_ids: torch.Tensor,
        spk_ids: torch.Tensor,
    ):
        with torch.no_grad():
            # Each decoded obj contains text and y_sequence - not collapsed seq.
            # To get collapsed seq tokens, the easiest hack is to tokenize the text back to ids.
            hyp_ids = []
            hyp_lens = []
            word_timestamps = []
            word_timestamps_lengths = []
            try:
                decoded = self.decode(predictions, predictions_lengths, None, None, None)

                hyp_ids = [torch.tensor(self.decoding.tokenizer.text_to_ids(x.text), dtype=torch.int32).to(predictions.device) for x in decoded]
                hyp_lens = [torch.tensor(len(x), dtype=torch.int32).to(predictions_lengths.device) for x in hyp_ids]

                for i, hyp in enumerate(decoded):
                    assert len(hyp.text.split()) == len(hyp.timestamp['word']), f"Number of words must match number of word timestamps: {len(hyp.text.split())} != {len(hyp.timestamp['word'])}"

                    if not hyp.text:
                        word_timestamps.append(torch.tensor([], dtype=torch.int32, device=predictions.device))
                        word_timestamps_lengths.append(torch.tensor(0, dtype=torch.int32, device=predictions.device))
                    else:
                        word_timestamps.append(torch.tensor([[w['start_offset'], w['end_offset']] for w in hyp.timestamp['word']], dtype=torch.int32, device=predictions.device))
                        word_timestamps_lengths.append(torch.tensor(len(hyp.timestamp['word']), dtype=torch.int32, device=predictions.device))

            except Exception as e:
                hyp_ids = []
                hyp_lens = []
                word_timestamps = []
                word_timestamps_lengths = []

                logging.error(f"MeetevalMTWER (1): {e}")
                # There's some fucked-up example without any offsets. It's some NeMo bug.
                # An easy workaround is to skip these samples (pretend as if '' was decoded)and just shoot a warning.
                logging.warning(f"MeetevalMTWER: Error decoding predictions. Falling back to decoding one-by-one and skipping the faulty inputs.")
                for i in range(len(predictions)):
                    try:
                        decoded = self.decode(predictions[i:i+1], predictions_lengths[i:i+1], None, None, None)
                        hyp_id = torch.tensor(self.decoding.tokenizer.text_to_ids(decoded[0].text), dtype=torch.int32).to(predictions.device)
                        hyp_len = torch.tensor(len(decoded[0].text), dtype=torch.int32).to(predictions_lengths.device)

                        if not decoded[0].text:
                            word_timestamp = torch.tensor([], dtype=torch.int32, device=predictions.device)
                            word_timestamps_length = torch.tensor(0, dtype=torch.int32, device=predictions.device)
                        else:
                            assert len(decoded[0].text.split()) == len(decoded[0].timestamp['word']), f"(2) Number of words must match number of word timestamps: {len(decoded[0].text.split())} != {len(decoded[0].timestamp['word'])}"
                            word_timestamp = torch.tensor([[w['start_offset'], w['end_offset']] for w in decoded[0].timestamp['word']], dtype=torch.int32, device=predictions.device)
                            word_timestamps_length = torch.tensor(len(decoded[0].timestamp['word']), dtype=torch.int32, device=predictions.device)
                    except Exception as e:
                        logging.error(f"MeetevalMTWER (2): {e}")
                        logging.warning(f"MeetevalMTWER: Error decoding predictions. Falling back to decoding one-by-one and skipping the faulty inputs.")
                        hyp_id = torch.tensor([], dtype=torch.int32, device=predictions.device)
                        hyp_len = torch.tensor(0, dtype=torch.int32, device=predictions.device)
                        word_timestamp = torch.tensor([], dtype=torch.int32, device=predictions.device)
                        word_timestamps_length = torch.tensor(0, dtype=torch.int32, device=predictions.device)

                    hyp_ids.append(hyp_id)
                    hyp_lens.append(hyp_len)
                    word_timestamps.append(word_timestamp)
                    word_timestamps_lengths.append(word_timestamps_length)

            # Ensure we store all utterances, even the faulty ones so meeteval doesn't complain during scoring about missing hypotheses..
            assert len(hyp_ids) == len(hyp_lens) == len(word_timestamps) == len(word_timestamps_lengths) == len(predictions) == len(utt_ids) == len(spk_ids), f"Number of decoded hypotheses must match number of predictions: {len(hyp_ids)} != {len(hyp_lens)} != {len(word_timestamps)} != {len(word_timestamps_lengths)} != {len(predictions)} != {len(utt_ids)} != {len(spk_ids)}"

            self.preds.extend(hyp_ids)
            self.preds_lengths.extend(hyp_lens)
            self.preds_word_timestamps.extend(word_timestamps)
            self.preds_word_timestamps_lengths.extend(word_timestamps_lengths)
            self.utt_ids.extend(utt_ids.detach())
            self.spk_ids.extend(spk_ids.detach())

    @staticmethod
    def _process_metric_res(res):
        output = {'wer': 0, 'ins': 0, 'del': 0, 'sub': 0, 'len': 0}
        for i in res:
            output['len'] += res[i].length
            output['ins'] += res[i].insertions
            output['del'] += res[i].deletions
            output['sub'] += res[i].substitutions
        output['wer'] = (output['sub'] + output['ins'] + output['del']) / output['len']
        return output

    @staticmethod
    def _reduce_res(res_all_ranks: List[Dict]):
        res = {k: 0 for k in res_all_ranks[0]}
        for res_rank in res_all_ranks:
            for k in res:
                res[k] += res_rank[k]
        return res

    def compute(self, targets_collection: List[Dict]):
        preds = dim_zero_cat(self.preds)
        preds_lengths = dim_zero_cat(self.preds_lengths)
        preds_word_timestamps = dim_zero_cat(self.preds_word_timestamps)
        preds_word_timestamps_lengths = dim_zero_cat(self.preds_word_timestamps_lengths)
        utt_ids = dim_zero_cat(self.utt_ids)
        spk_ids = dim_zero_cat(self.spk_ids)

        gt_segments = dict()
        for i in range(len(targets_collection)):
            speakers = sorted(list(set(x['speaker'] for x in targets_collection[i].text_tokens)))
            speaker_to_idx = {s: i for i, s in enumerate(speakers)}
            for seg in targets_collection[i].text_tokens:
                if seg['speaker'] not in speaker_to_idx:
                    raise Exception(f"Speaker {seg['speaker']} not found in {speakers}")

                if targets_collection[i].id not in gt_segments:
                    gt_segments[targets_collection[i].id] = []

                gt_segments[targets_collection[i].id].append(SegLstSegment(session_id=targets_collection[i].id, 
                                                 speaker=speaker_to_idx[seg['speaker']], 
                                                 words=self.decoding.decode_tokens_to_str(seg['text']), 
                                                 start_time=seg['start'],
                                                 end_time=seg['start'] + seg['duration']))
        gt_segment_ids = sorted(list(gt_segments.keys()))

        # It might happen that when running validation using multiple GPUS, # of samples is not divisible by # of GPUS => some exapmles are duplicated (batch padding).
        # Hence, we need to keep track of already processed pairs (utt_id, spk_id) to avoid double counting some errors.
        already_processed_pairs = set()
        pred_segments = dict()
        current_start = 0
        current_ts_start = 0
        for i in range(len(preds_lengths)):
            if (utt_ids[i].item(), spk_ids[i].item()) in already_processed_pairs:
                current_start += preds_lengths[i]
                current_ts_start += preds_word_timestamps_lengths[i]
                continue

            if utt_ids[i].item() not in pred_segments:
                pred_segments[utt_ids[i].item()] = []

            if preds_lengths[i] == 0:
                assert preds_word_timestamps_lengths[i] == 0, f"Number of word timestamps must be 0 if number of tokens is 0: {preds_word_timestamps_lengths[i]}"
                pred_segments[utt_ids[i].item()].append(SegLstSegment(session_id=utt_ids[i].item(), 
                                                   speaker=spk_ids[i].item(), 
                                                   words='', 
                                                   start_time=0, 
                                                   end_time=1))
                current_start += preds_lengths[i]
                current_ts_start += preds_word_timestamps_lengths[i]
                continue
            
            already_processed_pairs.add((utt_ids[i].item(), spk_ids[i].item()))
            current_transcript = self.decoding.decode_tokens_to_str(preds[current_start:current_start + preds_lengths[i]].detach().cpu())
            words = current_transcript.split()
            if not words:
                pred_segments[utt_ids[i].item()].append(SegLstSegment(session_id=utt_ids[i].item(), 
                                                   speaker=spk_ids[i].item(), 
                                                   words='', 
                                                   start_time=0, 
                                                   end_time=1))
                current_start += preds_lengths[i]
                current_ts_start += preds_word_timestamps_lengths[i]
                continue

            word_timestamps = preds_word_timestamps[current_ts_start:current_ts_start + preds_word_timestamps_lengths[i].detach().cpu()]
            assert len(word_timestamps) == len(words), f"Number of word timestamps must match number of words: {len(word_timestamps)} != {len(words)}, {preds_lengths}, {preds_word_timestamps_lengths}"

            current_start += preds_lengths[i]
            current_ts_start += preds_word_timestamps_lengths[i]

            if self.output_per_word_timestamps:
                current_segment_words = [(words[0], *word_timestamps[0])]
                last_word_end = word_timestamps[0][1]
                for j in range(len(words)):
                    pred_segments[utt_ids[i].item()].append(SegLstSegment(session_id=utt_ids[i].item(), 
                                                       speaker=spk_ids[i].item(), 
                                                       words=words[j], 
                                                       start_time=word_timestamps[j][0] * self.embed_duration, 
                                                       end_time=word_timestamps[j][1] * self.embed_duration))
            else:
                current_segment_words = [(words[0], *word_timestamps[0])]
                last_word_end = word_timestamps[0][1]
                for j in range(len(words)):
                    if (word_timestamps[j][0] - last_word_end)*self.embed_duration > 0.5:
                        pred_segments[utt_ids[i].item()].append(SegLstSegment(session_id=utt_ids[i].item(), 
                                                speaker=spk_ids[i].item(), 
                                                words=' '.join([w[0] for w in current_segment_words]), 
                                                start_time=current_segment_words[0][1] * self.embed_duration, 
                                                end_time=current_segment_words[-1][2] * self.embed_duration))
                        current_segment_words = [(words[j], *word_timestamps[j])]
                        last_word_end = word_timestamps[j][1]
                    else:
                        current_segment_words.append((words[j], *word_timestamps[j]))
                        last_word_end = word_timestamps[j][1]

                if len(current_segment_words) > 0:
                    pred_segments[utt_ids[i].item()].append(SegLstSegment(session_id=utt_ids[i].item(), 
                                                    speaker=spk_ids[i].item(), 
                                                    words=' '.join([w[0] for w in current_segment_words]), 
                                                    start_time=current_segment_words[0][1] * self.embed_duration, 
                                                    end_time=current_segment_words[-1][2] * self.embed_duration))

        pred_segment_ids = sorted(list(pred_segments.keys()))
        
        assert len(gt_segment_ids) == len(pred_segment_ids), f"Number of gt segments must match number of pred segments: {len(gt_segment_ids)} != {len(pred_segment_ids)}"

        
        num_elems_per_rank = ceil(len(gt_segment_ids) / get_world_size())
        begin_idx = get_rank() * num_elems_per_rank
        end_idx = (get_rank() + 1) * num_elems_per_rank
        gt_seg_lst = SegLST(segments=[seg for uid in gt_segment_ids[begin_idx:end_idx] for seg in gt_segments[uid]])
        pred_seg_lst = SegLST(segments=[seg for uid in pred_segment_ids[begin_idx:end_idx] for seg in pred_segments[uid]])

        res_cp = self._process_metric_res(meeteval.wer.cpwer(reference=gt_seg_lst, hypothesis=pred_seg_lst))
        res_tcp = self._process_metric_res(meeteval.wer.tcpwer(reference=gt_seg_lst, hypothesis=pred_seg_lst, collar=5))

        res_both_all_ranks = [None] * get_world_size()
        if get_world_size() > 1:
            torch.distributed.all_gather_object(res_both_all_ranks, (res_cp, res_tcp))
        else:
            res_both_all_ranks[0] = (res_cp, res_tcp)

        res_cp = self._reduce_res([res[0] for res in res_both_all_ranks])
        res_tcp = self._reduce_res([res[1] for res in res_both_all_ranks])

        return res_cp, res_tcp
