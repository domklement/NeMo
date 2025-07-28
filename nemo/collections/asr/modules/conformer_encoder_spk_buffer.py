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

import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import torch
import torch.distributed
from omegaconf import DictConfig, ListConfig, open_dict
from torch import nn

from nemo.collections.asr.models.configs import CacheAwareStreamingConfig
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
from nemo.collections.asr.parts.submodules.causal_convs import CausalConv1D
from nemo.collections.asr.parts.submodules.conformer_modules import ConformerLayer
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.parts.submodules.multi_head_attention import (
    LocalAttRelPositionalEncoding,
    MultiHeadAttention,
    NoPositionalEncoding,
    PositionalEncoding,
    RelPositionalEncoding,
    RelPositionMultiHeadAttention,
    RelPositionMultiHeadAttentionLongformer,
)
from nemo.collections.asr.parts.submodules.subsampling import (
    ConvSubsampling,
    StackingSubsampling,
    SubsamplingReductionModule,
)
from nemo.collections.asr.parts.utils import adapter_utils
from nemo.collections.asr.parts.utils.regularization_utils import compute_stochastic_depth_drop_probs
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import AccessMixin, adapter_mixins
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    BoolType,
    ChannelType,
    LengthsType,
    NeuralType,
    SpectrogramType,
)
from nemo.utils import logging

__all__ = ['ConformerEncoderSpkBuff']


class ConformerEncoderSpkBuff(ConformerEncoder):
    """
    The encoder for ASR model of Conformer.
    Based on this paper:
    'Conformer: Convolution-augmented Transformer for Speech Recognition' by Anmol Gulati et al.
    https://arxiv.org/abs/2005.08100

    Args:
        feat_in (int): the size of feature channels
        n_layers (int): number of layers of ConformerBlock
        d_model (int): the hidden size of the model
        feat_out (int): the size of the output features
            Defaults to -1 (means feat_out is d_model)
        subsampling (str): the method of subsampling:
            choices = ['vggnet', 'striding', 'dw-striding', 'stacking', 'stacking_norm']
            Defaults to striding.
        subsampling_factor (int): the subsampling factor which should be power of 2
            Defaults to 4.
        subsampling_conv_chunking_factor(int): optionally, force chunk inputs (helpful for large inputs)
            Should be power of 2, 1 (auto-chunking, default), or -1 (no chunking)
        subsampling_conv_channels (int): the size of the convolutions in the subsampling module
            Defaults to -1 which would set it to d_model.
        reduction (str, Optional): the method of reduction, choices=['pooling', 'striding']. If no value
            is passed, then no reduction is performed and the models runs with the original 4x subsampling.
        reduction_position (int, Optional): the index of the layer to apply reduction. If -1, apply reduction
            at the end.
        reduction_factor (int): the reduction factor which should be either 1 or a power of 2
            Defaults to 1.
        ff_expansion_factor (int): the expansion factor in feed forward layers
            Defaults to 4.
        self_attention_model (str): the type of the attention layer and positional encoding.

            'rel_pos':
                relative positional embedding and Transformer-XL
            'rel_pos_local_attn':
                relative positional embedding and Transformer-XL with local attention using
                overlapping chunks. Attention context is determined by att_context_size parameter.
            'abs_pos':
                absolute positional embedding and Transformer

            Default is rel_pos.
        pos_emb_max_len (int): the maximum length of positional embeddings
            Defaults to 5000
        n_heads (int): number of heads in multi-headed attention layers
            Defaults to 4.
        att_context_size (List[Union[List[int],int]]): specifies the context sizes on each side.
            Each context size should be a list of two integers like `[100, 100]`.
            A list of context sizes like `[[100,100]`, `[100,50]]` can also be passed. -1 means unlimited context.
            Defaults to `[-1, -1]`
        att_context_probs (List[float]): a list of probabilities of each one of the att_context_size
            when a list of them is passed. If not specified, uniform distribution is being used.
            Defaults to None
        att_context_style (str): 'regular' or 'chunked_limited'.
            Defaults to 'regular'
        xscaling (bool): enables scaling the inputs to the multi-headed attention layers by `sqrt(d_model)`.
            Defaults to True.
        untie_biases (bool): whether to not share (untie) the bias weights between layers of Transformer-XL
            Defaults to True.
        conv_kernel_size (int): the size of the convolutions in the convolutional modules
            Defaults to 31.
        conv_norm_type (str): the type of the normalization in the convolutional modules
            Defaults to 'batch_norm'.
        conv_context_size (list): it can be"causal" or a list of two integers
            while `conv_context_size[0]+conv_context_size[1]+1==conv_kernel_size`.
            `None` means `[(conv_kernel_size-1)//2`, `(conv_kernel_size-1)//2]`, and 'causal' means
            `[(conv_kernel_size-1), 0]`.
            Defaults to None.
        conv_dual_mode (bool): specifies if convolution should be dual mode when dual_offline mode is being used.
            When enables, the left half of the convolution kernel would get masked in streaming cases.
            Defaults to False.
        use_bias (bool): Use bias in all Linear and Conv1d layers from each ConformerLayer to improve
            activation flow and stabilize training of huge models.
            Defaults to True.
        dropout (float): the dropout rate used in all layers except the attention layers
            Defaults to 0.1.
        dropout_pre_encoder (float): the dropout rate used before the encoder
            Defaults to 0.1.
        dropout_emb (float): the dropout rate used for the positional embeddings
            Defaults to 0.1.
        dropout_att (float): the dropout rate used for the attention layer
            Defaults to 0.0.
        stochastic_depth_drop_prob (float): if non-zero, will randomly drop
            layers during training. The higher this value, the more often layers
            are dropped. Defaults to 0.0.
        stochastic_depth_mode (str): can be either "linear" or "uniform". If
            set to "uniform", all layers have the same probability of drop. If
            set to "linear", the drop probability grows linearly from 0 for the
            first layer to the desired value for the final layer. Defaults to
            "linear".
        stochastic_depth_start_layer (int): starting layer for stochastic depth.
            All layers before this will never be dropped. Note that drop
            probability will be adjusted accordingly if mode is "linear" when
            start layer is > 1. Defaults to 1.
        global_tokens (int): number of tokens to be used for global attention.
            Only relevant if self_attention_model is 'rel_pos_local_attn'.
            Defaults to 0.
        global_tokens_spacing (int): how far apart the global tokens are
            Defaults to 1.
        global_attn_separate (bool): whether the q, k, v layers used for global tokens should be separate.
            Defaults to False.
        use_pytorch_sdpa (bool): use torch sdpa instead of manual attention.
            Defaults to False.
        use_pytorch_sdpa_backends (list[str]): list of backend names to use in sdpa.
            None or empty list means all backends. e.g. ["MATH"]
            Defaults to None.
        bypass_pre_encode: if True, skip the pre-encoder module and the `audio_signal` should be pre-encoded
            embeddings. The `audio_signal` input supports two formats depending on the `bypass_pre_encode`
            boolean flag. This determines the required format of the input variable `audio_signal`.
            Defaults to `bypass_pre_encode=False`. `bypass_pre_encode=True` is used for the cases
            where frame-level, context-independent embeddings are needed to be saved or reused.
            (e.g., speaker cache in streaming speaker diarization)
        sync_max_audio_length (bool): when true, performs NCCL all_reduce to allocate the same amount of memory for
            positional encoding buffers on all GPUs. Disabling this setting may help with deadlocks in certain
            scenarios such as model parallelism, or generally when this module is not being ran on some GPUs
            as a part of the training step.
    """

    def __init__(
        self,
        feat_in,
        n_layers,
        d_model,
        feat_out=-1,
        causal_downsampling=False,
        subsampling='striding',
        subsampling_factor=4,
        subsampling_conv_chunking_factor=1,
        subsampling_conv_channels=-1,
        reduction=None,
        reduction_position=None,
        reduction_factor=1,
        ff_expansion_factor=4,
        self_attention_model='rel_pos',
        n_heads=4,
        att_context_size=None,
        att_context_probs=None,
        att_context_style='regular',
        xscaling=True,
        untie_biases=True,
        pos_emb_max_len=5000,
        conv_kernel_size=31,
        conv_norm_type='batch_norm',
        conv_context_size=None,
        use_bias=True,
        dropout=0.1,
        dropout_pre_encoder=0.1,
        dropout_emb=0.1,
        dropout_att=0.0,
        stochastic_depth_drop_prob: float = 0.0,
        stochastic_depth_mode: str = "linear",
        stochastic_depth_start_layer: int = 1,
        global_tokens: int = 0,
        global_tokens_spacing: int = 1,
        global_attn_separate: bool = False,
        use_pytorch_sdpa: bool = False,
        use_pytorch_sdpa_backends=None,
        sync_max_audio_length: bool = True,
        prepend_global_tokens: bool = False, # If True, add trainable tokens to the beginning of the sequence
        use_ce_spk_buffer: bool = False, # If True, use the CE speaker buffer as well at the end of the ConformerLayer.
        use_ce_spk_buffer_bias: bool = True, # If True, use bias in the CE speaker buffer.
        share_spk_buffer: bool = False, # If True, share the speaker buffer between the ConformerLayer and the CE speaker buffer.
        spk_buffer_size: int = 128,
    ):
        super().__init__(
            feat_in=feat_in,
            n_layers=n_layers,
            d_model=d_model,
            feat_out=feat_out,
            causal_downsampling=causal_downsampling,
            subsampling=subsampling,
            subsampling_factor=subsampling_factor,
            subsampling_conv_chunking_factor=subsampling_conv_chunking_factor,
            subsampling_conv_channels=subsampling_conv_channels,
            reduction=reduction,
            reduction_position=reduction_position,
            reduction_factor=reduction_factor,
            ff_expansion_factor=ff_expansion_factor,
            self_attention_model=self_attention_model,
            n_heads=n_heads,
            att_context_size=att_context_size,
            att_context_probs=att_context_probs,
            att_context_style=att_context_style,
            xscaling=xscaling,
            untie_biases=untie_biases,
            pos_emb_max_len=pos_emb_max_len,
            conv_kernel_size=conv_kernel_size,
            conv_norm_type=conv_norm_type,
            conv_context_size=conv_context_size,
            use_bias=use_bias,
            dropout=dropout,
            dropout_pre_encoder=dropout_pre_encoder,
            dropout_emb=dropout_emb,
            dropout_att=dropout_att,
            stochastic_depth_drop_prob=stochastic_depth_drop_prob,
            stochastic_depth_mode=stochastic_depth_mode,
            stochastic_depth_start_layer=stochastic_depth_start_layer,
            global_tokens=global_tokens,
            global_tokens_spacing=global_tokens_spacing,
            global_attn_separate=global_attn_separate,
            use_pytorch_sdpa=use_pytorch_sdpa,
            use_pytorch_sdpa_backends=use_pytorch_sdpa_backends,
            sync_max_audio_length=sync_max_audio_length,
        )

        self.n_heads = n_heads
        self.dropout_att = dropout_att

        self.prepend_global_tokens = prepend_global_tokens
        self.use_ce_spk_buffer = use_ce_spk_buffer
        self.spk_buffer_size = spk_buffer_size
        self.share_spk_buffer = share_spk_buffer
        self.use_ce_spk_buffer_bias = use_ce_spk_buffer_bias

        if self.prepend_global_tokens:
            assert self.global_tokens_spacing == 1, "global_tokens_spacing must be 1 when prepend_global_tokens is True"
            self.extra_global_tokens = nn.Parameter(torch.randn(self.global_tokens, self.d_model))

        if self.use_ce_spk_buffer:
            if self.share_spk_buffer:
                self.spk_buffer = nn.Parameter(torch.randn(self.spk_buffer_size, self.d_model))
            else:
                self.spk_buffer = nn.ParameterList([
                    nn.Parameter(torch.randn(self.spk_buffer_size, self.d_model))
                    for _ in range(self.n_layers)
                ])

            self.spk_buffer_ce = nn.ModuleList([
                MultiHeadAttention(
                    n_head=self.n_heads,
                    n_feat=self.d_model,
                    dropout_rate=self.dropout_att,
                    use_pytorch_sdpa=self.use_pytorch_sdpa,
                    use_pytorch_sdpa_backends=self.use_pytorch_sdpa_backends,
                    use_bias=self.use_ce_spk_buffer_bias, # The layer is followed by a LayerNorm, so no bias is needed (technically).
                )
                for _ in range(self.n_layers)
            ])
            self.spk_buffer_ln = nn.ModuleList([
                nn.LayerNorm(self.d_model)
                for _ in range(self.n_layers)
            ])
            self.spk_buffer_out_ln = nn.ModuleList([
                nn.LayerNorm(self.d_model)
                for _ in range(self.n_layers)
            ])
        # print("spk_buffer", self.spk_buffer)

    @typecheck()
    def forward(
        self,
        audio_signal,
        length,
        cache_last_channel=None,
        cache_last_time=None,
        cache_last_channel_len=None,
        bypass_pre_encode=False,
        output_prepend_global_tokens=False,
    ):
        """
        Forward function for the ConformerEncoder accepting an audio signal and its corresponding length.
        The `audio_signal` input supports two formats depending on the `bypass_pre_encode` boolean flag.
        This determines the required format of the input variable `audio_signal`:
        (1) bypass_pre_encode = False (default):
            `audio_signal` must be a tensor containing audio features.
            Shape: (batch, self._feat_in, n_frames)
        (2) bypass_pre_encode = True:
            `audio_signal` must be a tensor containing pre-encoded embeddings.
            Shape: (batch, n_frame, self.d_model)
        """
        if not bypass_pre_encode and audio_signal.shape[-2] != self._feat_in:
            raise ValueError(
                f"If bypass_pre_encode is False, audio_signal should have shape "
                f"(batch, {self._feat_in}, n_frame) but got last dimension {audio_signal.shape[-2]}."
            )
        if bypass_pre_encode and audio_signal.shape[-1] != self.d_model:
            raise ValueError(
                f"If bypass_pre_encode is True, audio_signal should have shape "
                f"(batch, n_frame, {self.d_model}) but got last dimension {audio_signal.shape[-1]}."
            )

        if bypass_pre_encode:
            self.update_max_seq_length(
                seq_length=audio_signal.size(2) * self.subsampling_factor, device=audio_signal.device
            )
        else:
            self.update_max_seq_length(seq_length=audio_signal.size(2), device=audio_signal.device)
        return self.forward_internal(
            audio_signal,
            length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            bypass_pre_encode=bypass_pre_encode,
            output_prepend_global_tokens=output_prepend_global_tokens,
        )

    def forward_internal(
        self,
        audio_signal,
        length,
        cache_last_channel=None,
        cache_last_time=None,
        cache_last_channel_len=None,
        bypass_pre_encode=False,
        output_prepend_global_tokens=False,
    ):
        # print(self.spk_buffer[0], self.extra_global_tokens)
        """
        The `audio_signal` input supports two formats depending on the `bypass_pre_encode` boolean flag.
        This determines the required format of the input variable `audio_signal`:
        (1) bypass_pre_encode = False (default):
            `audio_signal` must be a tensor containing audio features.
            Shape: (batch, self._feat_in, n_frames)
        (2) bypass_pre_encode = True:
            `audio_signal` must be a tensor containing pre-encoded embeddings.
            Shape: (batch, n_frame, self.d_model)

        `bypass_pre_encode=True` is used in cases where frame-level, context-independent embeddings are
        needed to be saved or reused (e.g., speaker cache in streaming speaker diarization).
        """
        if length is None:
            length = audio_signal.new_full(
                (audio_signal.size(0),), audio_signal.size(-1), dtype=torch.int64, device=audio_signal.device
            )

        # select a random att_context_size with the distribution specified by att_context_probs during training
        # for non-validation cases like test, validation or inference, it uses the first mode in self.att_context_size
        if self.training and len(self.att_context_size_all) > 1:
            cur_att_context_size = random.choices(self.att_context_size_all, weights=self.att_context_probs)[0]
        else:
            cur_att_context_size = self.att_context_size

        if not bypass_pre_encode:
            audio_signal = torch.transpose(audio_signal, 1, 2)

            if isinstance(self.pre_encode, nn.Linear):
                audio_signal = self.pre_encode(audio_signal)
            else:
                audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
                length = length.to(torch.int64)
                # `self.streaming_cfg` is set by setup_streaming_cfg(), called in the init
                if self.streaming_cfg.drop_extra_pre_encoded > 0 and cache_last_channel is not None:
                    audio_signal = audio_signal[:, self.streaming_cfg.drop_extra_pre_encoded :, :]
                    length = (length - self.streaming_cfg.drop_extra_pre_encoded).clamp(min=0)

            if self.reduction_position is not None and cache_last_channel is not None:
                raise ValueError("Caching with reduction feature is not supported yet!")

        max_audio_length = audio_signal.size(1)
        if cache_last_channel is not None:
            cache_len = self.streaming_cfg.last_channel_cache_size
            cache_keep_size = max_audio_length - self.streaming_cfg.cache_drop_size
            max_audio_length = max_audio_length + cache_len
            padding_length = length + cache_len
            offset = torch.neg(cache_last_channel_len) + cache_len
        else:
            padding_length = length
            cache_last_channel_next = None
            cache_len = 0
            offset = None

        audio_signal, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)

        # Create the self-attention and padding masks
        pad_mask, att_mask = self._create_masks(
            att_context_size=cur_att_context_size,
            padding_length=padding_length,
            max_audio_length=max_audio_length,
            offset=offset,
            device=audio_signal.device,
        )

        if self.prepend_global_tokens:
            assert att_mask is None
            bsize = audio_signal.shape[0]
            audio_signal = torch.cat([self.extra_global_tokens.unsqueeze(0).repeat(bsize, 1, 1), audio_signal], dim=1)
            pad_mask = torch.cat([torch.zeros(bsize, self.global_tokens, dtype=torch.bool, device=audio_signal.device), pad_mask], dim=1)

        if cache_last_channel is not None:
            pad_mask = pad_mask[:, cache_len:]
            if att_mask is not None:
                att_mask = att_mask[:, cache_len:]
            # Convert caches from the tensor to list
            cache_last_time_next = []
            cache_last_channel_next = []

        for lth, (drop_prob, layer) in enumerate(zip(self.layer_drop_probs, self.layers)):
            original_signal = audio_signal
            if cache_last_channel is not None:
                cache_last_channel_cur = cache_last_channel[lth]
                cache_last_time_cur = cache_last_time[lth]
            else:
                cache_last_channel_cur = None
                cache_last_time_cur = None
            audio_signal = layer(
                x=audio_signal,
                att_mask=att_mask,
                pos_emb=pos_emb,
                pad_mask=pad_mask,
                cache_last_channel=cache_last_channel_cur,
                cache_last_time=cache_last_time_cur,
            )

            if self.use_ce_spk_buffer:
                if self.share_spk_buffer:
                    spk_buffer = self.spk_buffer.unsqueeze(0).repeat(audio_signal.shape[0], 1, 1)
                else:
                    spk_buffer = self.spk_buffer[lth].unsqueeze(0).repeat(audio_signal.shape[0], 1, 1)
                
                residual = audio_signal
                audio_signal = self.spk_buffer_ln[lth](audio_signal)
                audio_signal = self.spk_buffer_ce[lth](audio_signal, spk_buffer, spk_buffer, mask=None) + residual
                audio_signal = self.spk_buffer_out_ln[lth](audio_signal)

            if cache_last_channel_cur is not None:
                (audio_signal, cache_last_channel_cur, cache_last_time_cur) = audio_signal
                cache_last_channel_next.append(cache_last_channel_cur)
                cache_last_time_next.append(cache_last_time_cur)

            # applying stochastic depth logic from https://arxiv.org/abs/2102.03216
            if self.training and drop_prob > 0.0:
                should_drop = torch.rand(1) < drop_prob
                # adjusting to match expectation
                if should_drop:
                    # that's not efficient, but it's hard to implement distributed
                    # version of dropping layers without deadlock or random seed meddling
                    # so multiplying the signal by 0 to ensure all weights get gradients
                    audio_signal = audio_signal * 0.0 + original_signal
                else:
                    # not doing this operation if drop prob is 0 as it's identity in that case
                    audio_signal = (audio_signal - original_signal) / (1.0 - drop_prob) + original_signal

            if self.reduction_position == lth:
                audio_signal, length = self.reduction_subsampling(x=audio_signal, lengths=length)
                max_audio_length = audio_signal.size(1)
                # Don't update the audio_signal here because then it will again scale the audio_signal
                # and cause an increase in the WER
                _, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)
                pad_mask, att_mask = self._create_masks(
                    att_context_size=cur_att_context_size,
                    padding_length=length,
                    max_audio_length=max_audio_length,
                    offset=offset,
                    device=audio_signal.device,
                )

            # saving tensors if required for interctc loss
            if self.is_access_enabled(getattr(self, "model_guid", None)):
                if self.interctc_capture_at_layers is None:
                    self.interctc_capture_at_layers = self.access_cfg.get('interctc', {}).get('capture_layers', [])
                if lth in self.interctc_capture_at_layers:
                    lth_audio_signal = audio_signal
                    if self.out_proj is not None:
                        lth_audio_signal = self.out_proj(audio_signal)
                    # shape is the same as the shape of audio_signal output, i.e. [B, D, T]
                    self.register_accessible_tensor(
                        name=f'interctc/layer_output_{lth}', tensor=torch.transpose(lth_audio_signal, 1, 2)
                    )
                    self.register_accessible_tensor(name=f'interctc/layer_length_{lth}', tensor=length)

        if output_prepend_global_tokens:
            if self.prepend_global_tokens:
                global_tokens_out = audio_signal[:, :self.global_tokens, :]
                audio_signal = audio_signal[:, self.global_tokens:, :]
                pad_mask = pad_mask[:, self.global_tokens:]
            else:
                global_tokens_out = audio_signal[:, :self.global_tokens:self.global_tokens_spacing, :]

        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal)

        # Reduction
        if self.reduction_position == -1:
            audio_signal, length = self.reduction_subsampling(x=audio_signal, lengths=length)

        audio_signal = torch.transpose(audio_signal, 1, 2)
        length = length.to(dtype=torch.int64)

        if output_prepend_global_tokens:
            return (
                audio_signal,
                length,
                global_tokens_out,
            )

        if cache_last_channel is not None:
            cache_last_channel_next = torch.stack(cache_last_channel_next, dim=0)
            cache_last_time_next = torch.stack(cache_last_time_next, dim=0)
            return (
                audio_signal,
                length,
                cache_last_channel_next,
                cache_last_time_next,
                torch.clamp(cache_last_channel_len + cache_keep_size, max=cache_len),
            )
        else:
            return audio_signal, length, None
        
    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
                "cache_last_channel": NeuralType(('D', 'B', 'T', 'D'), ChannelType(), optional=True),
                "cache_last_time": NeuralType(('D', 'B', 'D', 'T'), ChannelType(), optional=True),
                "cache_last_channel_len": NeuralType(tuple('B'), LengthsType(), optional=True),
                "bypass_pre_encode": NeuralType(tuple(), BoolType(), optional=True),
                "output_prepend_global_tokens": NeuralType(tuple(), BoolType(), optional=True),
            }
        )
    
    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return OrderedDict(
            {
                "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
                "global_tokens_out": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation(), optional=True),
                "cache_last_channel_next": NeuralType(('D', 'B', 'T', 'D'), ChannelType(), optional=True),
                "cache_last_time_next": NeuralType(('D', 'B', 'D', 'T'), ChannelType(), optional=True),
                "cache_last_channel_next_len": NeuralType(tuple('B'), LengthsType(), optional=True),
            }
        )



class ConformerEncoderSpkBuffAdapter(ConformerEncoderSpkBuff, adapter_mixins.AdapterModuleMixin):
    """This class inherits from ConformerEncoderSpkBuff and wraps the adapter mixin class."""

    # Higher level forwarding
    def add_adapter(self, name: str, cfg: dict):
        cfg = self._update_adapter_cfg_input_dim(cfg)
        for conformer_layer in self.layers:  # type: adapter_mixins.AdapterModuleMixin
            conformer_layer.add_adapter(name, cfg)

    def is_adapter_available(self) -> bool:
        return any([conformer_layer.is_adapter_available() for conformer_layer in self.layers])

    def set_enabled_adapters(self, name: Optional[str] = None, enabled: bool = True):
        for conformer_layer in self.layers:  # type: adapter_mixins.AdapterModuleMixin
            conformer_layer.set_enabled_adapters(name=name, enabled=enabled)

    def get_enabled_adapters(self) -> List[str]:
        names = set([])
        for conformer_layer in self.layers:  # type: adapter_mixins.AdapterModuleMixin
            names.update(conformer_layer.get_enabled_adapters())

        names = sorted(list(names))
        return names

    def _update_adapter_cfg_input_dim(self, cfg: DictConfig):
        cfg = adapter_utils.update_adapter_cfg_input_dim(self, cfg, module_dim=self.d_model)
        return cfg

    def get_accepted_adapter_types(
        self,
    ) -> Set[type]:
        types = super().get_accepted_adapter_types()

        if len(types) == 0:
            self.set_accepted_adapter_types(
                [
                    adapter_utils.LINEAR_ADAPTER_CLASSPATH,
                    adapter_utils.MHA_ADAPTER_CLASSPATH,
                    adapter_utils.RELMHA_ADAPTER_CLASSPATH,
                ]
            )
            types = self.get_accepted_adapter_types()
        return types


class ConformerMultiLayerFeatureExtractor(NeuralModule, Exportable, AccessMixin):
    """
    A wrapper module that extracts features from multiple layers of a ConformerEncoderSpkBuff,
    by reusing existing mechanisim for interctc loss.
    To use it, set `layer_idx_list` to  specify the indices of layers to extract from.
    Also, you can specify an `aggretator` module to aggregate the features from different layers,
    default not aggregating.
    """

    def __init__(
        self,
        encoder: ConformerEncoderSpkBuff,
        layer_idx_list: List[int],
        aggregator: NeuralModule = None,
        detach: bool = False,
        convert_to_cpu: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.layer_idx_list = [int(lyr_idx) for lyr_idx in layer_idx_list]
        for x in self.layer_idx_list:
            if x < 0 or x >= len(encoder.layers):
                raise ValueError(f"layer index {x} out of range [0, {len(encoder.layers)})")
        self.enc_access_cfg = {
            "interctc": {
                "capture_layers": self.layer_idx_list,
            },
            "detach": detach,
            "convert_to_cpu": convert_to_cpu,
        }
        self.aggregator = aggregator

    def forward(
        self, audio_signal, length, cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=missing-function-docstring
        old_access_flag = self.is_access_enabled(guid=getattr(self, "model_guid", None))
        self.update_access_cfg(self.enc_access_cfg, guid=getattr(self, "model_guid", None))
        self.set_access_enabled(access_enabled=True, guid=getattr(self, "model_guid", None))

        _ = self.encoder(
            audio_signal=audio_signal,
            length=length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )

        # Chunk of code adapted from ConformerEncoderSpkBuff.forward_internal()
        total_registry = {}
        for module_registry in self.get_module_registry(self.encoder).values():
            for key in module_registry:
                if key.startswith("interctc/") and key in total_registry:
                    raise RuntimeError(f"layer {key} has been logged multiple times!")
            total_registry.update(module_registry)

        encoded_list = []
        encoded_len_list = []
        for layer_idx in self.layer_idx_list:
            try:
                layer_outputs = total_registry[f"interctc/layer_output_{layer_idx}"]
                layer_lengths = total_registry[f"interctc/layer_length_{layer_idx}"]
            except KeyError:
                raise RuntimeError(
                    f"Intermediate layer {layer_idx} was not captured! "
                    "Check the layer index and the number of ConformerEncoderSpkBuff layers."
                )
            if len(layer_outputs) > 1 or len(layer_lengths) > 1:
                raise RuntimeError("Make sure encoder.forward is called exactly one time")
            encoded_list.append(layer_outputs[0])  # [B, D, T]
            encoded_len_list.append(layer_lengths[0])  # [B]

        self.encoder.reset_registry()
        self.set_access_enabled(access_enabled=old_access_flag, guid=getattr(self, "model_guid", None))
        # End of the adapted chunk

        if self.aggregator is not None:
            return self.aggregator(encoded_list, encoded_len_list)  # Tensor[B,D*L,T], Tensor[B]
        else:
            return encoded_list, encoded_len_list  # List[Tensor[B,D,T]], List[Tensor[B]]


# Register any additional information
if adapter_mixins.get_registered_adapter(ConformerEncoderSpkBuff) is None:
    adapter_mixins.register_adapter(base_class=ConformerEncoderSpkBuff, adapter_class=ConformerEncoderSpkBuffAdapter)


@dataclass
class ConformerChangeConfig:
    """
    Change self_attention_model for Conformer.

    Options:
     'rel_pos': relative positional embedding and Transformer-XL
     'rel_pos_local_attn': relative positional embedding and Transformer-XL with local attention using
      overlapping chunks. Attention context is determined by att_context_size parameter.
     'abs_pos': absolute positional embedding and Transformer
    """

    # If None is provided, self_attention_model is not changed.
    self_attention_model: Optional[str] = None

    # Change the attention context size by providing 2 integers,
    # corresponding to left and right context, or -1 for full context.
    # If None is provided, the attention context size isn't changed.
    att_context_size: Optional[List[int]] = None
