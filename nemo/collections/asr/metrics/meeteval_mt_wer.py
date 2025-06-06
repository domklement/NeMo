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

from typing import List, Optional, Tuple, Union

import meeteval
from meeteval.io.seglst import SegLstSegment, SegLST
import torch
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


from nemo.collections.asr.parts.submodules.ctc_decoding import AbstractCTCDecoding
from nemo.collections.asr.parts.submodules.multitask_decoding import AbstractMultiTaskDecoding
from nemo.collections.asr.parts.submodules.rnnt_decoding import AbstractRNNTDecoding
from nemo.utils import logging

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
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step, sync_on_compute=sync_on_compute)

        self.decoding = decoding
        self.use_cer = use_cer
        self.log_prediction = log_prediction
        self.fold_consecutive = fold_consecutive
        self.batch_dim_index = batch_dim_index

        self.decode = None
        if isinstance(self.decoding, AbstractRNNTDecoding):
            self.decode = lambda predictions, predictions_lengths, predictions_mask, input_ids, targets: self.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=predictions, encoded_lengths=predictions_lengths
            )
        elif isinstance(self.decoding, AbstractCTCDecoding):
            self.decode = lambda predictions, predictions_lengths, predictions_mask, input_ids, targets: self.decoding.ctc_decoder_predictions_tensor(
                decoder_outputs=predictions,
                decoder_lengths=predictions_lengths,
                fold_consecutive=self.fold_consecutive,
                return_hypotheses=False,
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
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        self.add_state("targets_lengths", default=[], dist_reduce_fx="cat")
        self.add_state("utt_ids", default=[], dist_reduce_fx="cat")
        self.add_state("spk_ids", default=[], dist_reduce_fx="cat")

    def update(
        self,
        predictions: torch.Tensor,
        predictions_lengths: torch.Tensor,
        targets: torch.Tensor,
        targets_lengths: torch.Tensor,
        utt_ids: torch.Tensor,
        spk_ids: torch.Tensor,
    ):
        with torch.no_grad():
            # Each decoded obj contains text and y_sequence - not collapsed seq.
            # To get collapsed seq tokens, the easiest hack is to tokenize the text back to ids.
            decoded = self.decode(predictions, predictions_lengths, None, None, None)
            hyp_ids = [torch.tensor(self.decoding.tokenizer.text_to_ids(x.text), dtype=torch.int32).to(predictions.device) for x in decoded]
            hyp_lens = [torch.tensor(len(x), dtype=torch.int32).to(predictions_lengths.device) for x in hyp_ids]

            for i, target in enumerate(targets):
                self.targets.append(
                    target[:targets_lengths[i]].detach()
                )

            self.preds.extend(hyp_ids)
            self.preds_lengths.extend(hyp_lens)
            # self.targets.extend(targets.detach())
            self.targets_lengths.extend(targets_lengths.detach())
            self.utt_ids.extend(utt_ids.detach())
            self.spk_ids.extend(spk_ids.detach())

    def compute(self):
        preds = dim_zero_cat(self.preds)
        preds_lengths = dim_zero_cat(self.preds_lengths)
        targets = dim_zero_cat(self.targets)
        targets_lengths = dim_zero_cat(self.targets_lengths)
        utt_ids = dim_zero_cat(self.utt_ids)
        spk_ids = dim_zero_cat(self.spk_ids)

        decoded_targets = []
        current_start = 0
        for i in range(len(targets_lengths)):
            decoded_targets.append(self.decoding.decode_tokens_to_str(targets[current_start:current_start + targets_lengths[i].detach().cpu()]))
            current_start += targets_lengths[i]

        decoded_preds = []
        current_start = 0
        for i in range(len(preds_lengths)):
            decoded_preds.append(self.decoding.decode_tokens_to_str(preds[current_start:current_start + preds_lengths[i]].detach().cpu()))
            current_start += preds_lengths[i]

        assert len(decoded_preds) == len(decoded_targets) == len(utt_ids) == len(spk_ids)

        # It might happen that when running validation using multiple GPUS, # of samples is not divisible by # of GPUS => some exapmles are duplicated (batch padding).
        # Hence, we need to keep track of already processed pairs (utt_id, spk_id) to avoid double counting some errors.
        already_processed_pairs = set()
        gt_segments = []
        for i in range(len(decoded_targets)):
            if (utt_ids[i].item(), spk_ids[i].item()) in already_processed_pairs:
                continue
            already_processed_pairs.add((utt_ids[i].item(), spk_ids[i].item()))
            gt_segments.append(SegLstSegment(session_id=utt_ids[i].item(), 
                                             speaker=spk_ids[i].item(), 
                                             words=decoded_targets[i], 
                                             start=0, 
                                             end=1))
        gt_segments = SegLST(segments=gt_segments)
        
        already_processed_pairs = set()
        pred_segments = []
        for i in range(len(decoded_preds)):
            if (utt_ids[i].item(), spk_ids[i].item()) in already_processed_pairs:
                continue
            already_processed_pairs.add((utt_ids[i].item(), spk_ids[i].item()))
            pred_segments.append(SegLstSegment(session_id=utt_ids[i].item(), 
                                               speaker=spk_ids[i].item(), 
                                               words=decoded_preds[i], 
                                               start=0, 
                                               end=1))
        pred_segments = SegLST(segments=pred_segments)

        res = meeteval.wer.cpwer(reference=gt_segments, hypothesis=pred_segments)
        
        length = 0
        insertions = 0
        deletions = 0
        substitutions = 0
        for i in res:
            length += res[i].length
            insertions += res[i].insertions
            deletions += res[i].deletions
            substitutions += res[i].substitutions

        return (insertions + deletions + substitutions) / length, insertions, deletions, substitutions, length
