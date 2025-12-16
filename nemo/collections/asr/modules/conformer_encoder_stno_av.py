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
from typing import List, Optional, Set, Tuple, Union

import torch
import torch.distributed
from omegaconf import DictConfig, ListConfig, open_dict
from torch import nn

from nemo.collections.asr.models.configs import CacheAwareStreamingConfig
from nemo.collections.asr.modules.fddt import FDDT
from nemo.collections.asr.modules.film import FiLM
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
from nemo.collections.asr.parts.submodules.causal_convs import CausalConv1D
from nemo.collections.asr.parts.submodules.conformer_modules import ConformerLayer
from nemo.collections.asr.modules.conformer_encoder_stno import ConformerEncoderSTNO
from nemo.collections.asr.parts.submodules.multi_head_attention import (
    LocalAttRelPositionalEncoding,
    MultiHeadAttention,
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
    MaskType,
    NeuralType,
    SpectrogramType,
)
from nemo.utils import logging

__all__ = ['ConformerEncoderSTNOAV']


class VisualProcessingModule(nn.Module):
    def __init__(self, d_visual_embeds, d_model, visual_downsampling_factor, visual_preprocessing_model='base', 
                 conditioning_embed_aggr_method='avg', 
                 num_conditioning_embeds=1):
        """

        """
        super().__init__()
        self.d_visual_embeds = d_visual_embeds
        self.d_model = d_model
        self.visual_downsampling_factor = visual_downsampling_factor
        self.visual_preprocessing_model = visual_preprocessing_model
        self.conditioning_embed_aggr_method = conditioning_embed_aggr_method
        self.num_conditioning_embeds = num_conditioning_embeds

        self.visual_ln = nn.LayerNorm(d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.visual_conv_downsampling = torch.nn.Conv1d(in_channels=d_visual_embeds, 
                                                        out_channels=d_model, 
                                                        kernel_size=5, 
                                                        stride=visual_downsampling_factor, 
                                                        padding=2)

        # Extra conditioning parameters
        if self.visual_preprocessing_model == 'extra_conv':
            self.extra_ln = nn.LayerNorm(d_model)
            self.extra_conv = torch.nn.Conv1d(in_channels=d_visual_embeds, 
                                              out_channels=d_model, 
                                              kernel_size=9, 
                                              stride=1, 
                                              padding=4)

        # Embedding aggregation parameters
        if conditioning_embed_aggr_method == 'wavg':
            self.log_weights = nn.Parameter(torch.ones(num_conditioning_embeds) / self.num_conditioning_embeds)

    def forward(self, visual_embeds, audio_signal):
        is_multispeaker = len(visual_embeds.shape) == 5
        if is_multispeaker:
            B_orig, T_orig, S_orig, C, D = visual_embeds.shape
            visual_embeds = visual_embeds.permute(0, 2, 1, 3, 4).reshape(B_orig*S_orig, T_orig, C, D)
        # visual_embeds: (B, T, C, D)
        if self.conditioning_embed_aggr_method == 'avg':
            visual_embeds = visual_embeds.mean(dim=2)
        elif self.conditioning_embed_aggr_method == 'wavg':
            weights = torch.exp(self.log_weights)
            visual_embeds = (visual_embeds * weights.view(1, 1, -1, 1)).sum(dim=2) / weights.sum()  # (B, T, D)
        else:
            raise ValueError(f'Unknown conditioning_embed_aggr_method: {self.conditioning_embed_aggr_method}')

        # visual_embeds: (B, T, D)
        B_v, T_v, D_v = visual_embeds.shape
        downsampled_visual_embeds = self.visual_conv_downsampling(
            visual_embeds.permute(0, 2, 1).reshape(B_v, D_v, T_v)
        ).reshape(B_v, self.d_model, -1).transpose(-1, -2)

        # Sometimes, the audio shape can be off-by-one. Fix it by either getting rid of or adding one frame.
        shape_diff = audio_signal.shape[1] - downsampled_visual_embeds.shape[1]
        if shape_diff == 1:
            downsampled_visual_embeds = torch.nn.functional.pad(downsampled_visual_embeds, (0,0,0,1,0,0))
        elif shape_diff == -1:
            downsampled_visual_embeds = downsampled_visual_embeds[:, :audio_signal.shape[1], :]
        elif shape_diff == 0:
            pass
        else:
            raise ValueError('Audio and visual embeddings have different time dimensions even after downsampling: {} vs {}.'.format(audio_signal.shape[1], downsampled_visual_embeds.shape[1]))
        
        # We have matching shapes between acoustic and conditioning sequence, now we can add more parameters to further transform the visual embeddings.
        if self.visual_preprocessing_model == 'extra_conv':
            downsampled_visual_embeds = self.extra_ln(downsampled_visual_embeds)
            downsampled_visual_embeds = self.extra_conv(
                downsampled_visual_embeds.permute(0, 2, 1)
            ).reshape(B_v, self.d_model, -1).transpose(-1, -2)

        if is_multispeaker:
            downsampled_visual_embeds = downsampled_visual_embeds.reshape(B_orig, S_orig, -1, self.d_model).permute(0, 2, 1, 3)
        
        return self.visual_ln(self.output_linear(downsampled_visual_embeds))
    

class ConcatAdapter(nn.Module):
    def __init__(self, d_model, bottleneck=128):
        super().__init__()
        self.norm_h = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)
        self.down = nn.Linear(2 * d_model, bottleneck)
        self.up   = nn.Linear(bottleneck, d_model)
        self.act  = nn.GELU()
        self.gate = nn.Parameter(torch.full((d_model,), -3.0))  # per-channel gate

    def forward(self, h, v):
        # h, v: (B, T, d_model), time-aligned
        h_n = self.norm_h(h)
        v_n = self.norm_v(v)
        x = torch.cat([h_n, v_n], dim=-1)
        z = self.act(self.down(x))
        delta = self.up(z)
        alpha = torch.sigmoid(self.gate).view(1, 1, -1)
        return h + alpha * delta


class VisualConditioningModule(nn.Module):
    def __init__(self, d_model, visual_conditioning_method, modality_dropout_prob=0.0, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.visual_conditioning_method = visual_conditioning_method
        self.modality_dropout_prob = modality_dropout_prob

        if visual_conditioning_method == 'add_gate':
            self.gate = nn.Parameter(torch.full((d_model,), -3.0))  # per-channel gate
        elif visual_conditioning_method == 'film':
            self.film_layer = FiLM(d_model)
        elif visual_conditioning_method == 'concat_add_gate':
            self.concat_adapter = ConcatAdapter(d_model)
        elif visual_conditioning_method == 'cross_attn':
            self.cross_attn = MultiHeadAttention(
                n_feat=d_model,
                n_head=8,
                dropout_rate=0.1,
            )
        elif visual_conditioning_method == 'rel_pos_cross_attn':
            self.cross_attn = RelPositionMultiHeadAttention(
                n_feat=d_model,
                n_head=8,
                dropout_rate=0.1,
                pos_bias_u=None,
                pos_bias_v=None,
                use_pytorch_sdpa=False,
                use_pytorch_sdpa_backends=None,
            )
            self.pos_enc = RelPositionalEncoding(
                d_model=d_model,
                dropout_rate=0.1,
                max_len=45000,
                xscale=True,
                dropout_rate_emb=0.1,
            )
            self.out_dropout = nn.Dropout(0.1)

            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            self.pos_enc.extend_pe(45000, device, dtype)

    def forward(self, audio_signal, visual_embeds, att_mask=None, **kwargs):
        if random.random() < self.modality_dropout_prob and self.training:
            audio_signal = 0 * audio_signal

        if self.visual_conditioning_method == 'add':
            conditioned_audio = audio_signal + visual_embeds
        elif self.visual_conditioning_method == 'add_gate':
            alpha = torch.sigmoid(self.gate).view(1, 1, -1)
            conditioned_audio = audio_signal + alpha * visual_embeds
        elif self.visual_conditioning_method == 'concat_add_gate':
            conditioned_audio = self.concat_adapter(audio_signal, visual_embeds)
        elif self.visual_conditioning_method == 'film':
            conditioned_audio = self.film_layer(audio_signal, visual_embeds)
        elif self.visual_conditioning_method == 'cross_attn':
            conditioned_audio = audio_signal + self.cross_attn(query=audio_signal, key=visual_embeds, value=visual_embeds, mask=att_mask)
        elif self.visual_conditioning_method == 'rel_pos_cross_attn':
            visual_embeds, pe = self.pos_enc(visual_embeds, cache_len=0)
            conditioned_audio = audio_signal + self.out_dropout(self.cross_attn(query=audio_signal, key=visual_embeds, value=visual_embeds, mask=att_mask, pos_emb=pe))
        else:
            raise ValueError(f'Unknown visual conditioning method: {self.visual_conditioning_method}')
        
        return conditioned_audio
    
class MultiSpeakerVisualConditioningModule(nn.Module):
    def __init__(self, d_model, visual_conditioning_method, modality_dropout_prob=0.0, max_num_speakers=8, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.visual_conditioning_method = visual_conditioning_method
        self.modality_dropout_prob = modality_dropout_prob
        self.max_num_speakers = max_num_speakers

        if visual_conditioning_method == 'add':
            self.tgt_proj = nn.Linear(d_model, d_model)
            self.nontgt_proj = nn.Linear(d_model, d_model)
            self.out_linear = nn.Linear(self.max_num_speakers * d_model, d_model)
        else:
            raise ValueError(f'Unknown visual conditioning method for multi-speaker: {self.visual_conditioning_method}')

    def forward(self, audio_signal, visual_embeds, num_speakers, att_mask=None):
        """
        visual_embeds: (B, T, S, D)
        """

        if self.visual_conditioning_method == 'add':
            # Project target and non-target embeddings
            tgt_embeds = self.tgt_proj(visual_embeds[:, :, :1, :])
            nontgt_embeds = self.nontgt_proj(visual_embeds[:, :, 1:, :])
            vis_embeds = torch.concat([tgt_embeds, nontgt_embeds], dim=2)
            vis_embeds = torch.nn.functional.pad(vis_embeds, (0, 0, 0, (self.max_num_speakers - vis_embeds.shape[2])))
            for i, n in enumerate(num_speakers):
                vis_embeds[i, :, n:, :] = 0.0  # Zero out embeddings for non-present speakers
            vis_embeds = self.out_linear(vis_embeds.reshape((*vis_embeds.shape[:2], -1)))  # (B, T, D)
            conditioned_audio = audio_signal + vis_embeds
        else:
            raise ValueError(f'Unknown visual conditioning method for multi-speaker: {self.visual_conditioning_method}')
        
        return conditioned_audio


class VisionAdapterEncoder(nn.Module):
    def __init__(self, d_model, num_layers=8, num_heads=8, d_ff_ratio=2, dropout=0.1, dropout_pre_enc=0.1):
        super().__init__()
        self.d_model = d_model
        self.pos_enc = RelPositionalEncoding(
            d_model=d_model,
            dropout_rate=dropout_pre_enc,
            max_len=45000,
            xscale=None,
            dropout_rate_emb=0.0,
        )
        self.layers = nn.ModuleList([
            ConformerLayer(
                d_model=d_model,
                d_ff=d_model * d_ff_ratio,
                self_attention_model='rel_pos',
                global_tokens=0,
                global_tokens_spacing=1,
                global_attn_separate=1,
                n_heads=num_heads,
                conv_kernel_size=31,
                conv_norm_type='batch_norm',
                conv_context_size=None,
                dropout=dropout,
                dropout_att=dropout,
                pos_bias_u=None,
                pos_bias_v=None,
                att_context_size=[-1, -1],
                use_bias=False,
                use_pytorch_sdpa=False,
                use_pytorch_sdpa_backends=None,
            ) for _ in range(num_layers)
        ])

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self.pos_enc.extend_pe(45000, device, dtype)
    
    def forward(self, x, att_mask=None, pad_mask=None):
        x, pos_emb = self.pos_enc(x, cache_len=0)
        for layer in self.layers:
            x = layer(x, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask)
        return x
        


class ConformerEncoderSTNOAV(ConformerEncoderSTNO):
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

    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        dev = next(self.parameters()).device
        if self.export_cache_support:
            window_size = max_dim
            if self.streaming_cfg is not None:
                if isinstance(self.streaming_cfg.chunk_size, list):
                    chunk_size = self.streaming_cfg.chunk_size[1]
                else:
                    chunk_size = self.streaming_cfg.chunk_size
                if isinstance(self.streaming_cfg.pre_encode_cache_size, list):
                    pre_encode_cache_size = self.streaming_cfg.pre_encode_cache_size[1]
                else:
                    pre_encode_cache_size = self.streaming_cfg.pre_encode_cache_size
                window_size = chunk_size + pre_encode_cache_size
            input_example = torch.randn(max_batch, self._feat_in, window_size, device=dev)
            input_example_length = torch.randint(
                window_size // 4, window_size, (max_batch,), device=dev, dtype=torch.int64
            )
            cache_last_channel, cache_last_time, cache_last_channel_len = self.get_initial_cache_state(
                batch_size=max_batch, device=dev, max_dim=max_dim
            )
            all_input_example = tuple(
                [
                    input_example,
                    input_example_length,
                    cache_last_channel.transpose(0, 1),
                    cache_last_time.transpose(0, 1),
                    cache_last_channel_len,
                ]
            )
        else:
            input_example = torch.randn(max_batch, self._feat_in, max_dim, device=dev)
            input_example_length = torch.randint(max_dim // 4, max_dim, (max_batch,), device=dev, dtype=torch.int64)
            all_input_example = tuple([input_example, input_example_length])

        return all_input_example

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
                "stno_mask": NeuralType(('B', 'S', 'T'), MaskType(), optional=True),
                "stno_mask_length": NeuralType(tuple('B'), LengthsType(), optional=True),
                "visual_embeds": NeuralType(('B', 'T', 'S', 'C', 'D'), SpectrogramType(), optional=True),
                "visual_embed_lengths": NeuralType(tuple('B'), LengthsType(), optional=True),
                "num_speakers": NeuralType(tuple('B'), LengthsType(), optional=True),
            }
        )

    @property
    def input_types_for_export(self):
        """Returns definitions of module input ports."""
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
                "cache_last_channel": NeuralType(('B', 'D', 'T', 'D'), ChannelType(), optional=True),
                "cache_last_time": NeuralType(('B', 'D', 'D', 'T'), ChannelType(), optional=True),
                "cache_last_channel_len": NeuralType(tuple('B'), LengthsType(), optional=True),
                "bypass_pre_encode": NeuralType(tuple(), BoolType(), optional=True),
                "stno_mask": NeuralType(('B', 'S', 'T'), MaskType(), optional=True),
                "stno_mask_length": NeuralType(tuple('B'), LengthsType(), optional=True),
                "visual_embeds": NeuralType(('B', 'T', 'S', 'C', 'D'), SpectrogramType(), optional=True),
                "visual_embed_lengths": NeuralType(tuple('B'), LengthsType(), optional=True),
                "num_speakers": NeuralType(tuple('B'), LengthsType(), optional=True),
            }
        )

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return OrderedDict(
            {
                "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
                "cache_last_channel_next": NeuralType(('D', 'B', 'T', 'D'), ChannelType(), optional=True),
                "cache_last_time_next": NeuralType(('D', 'B', 'D', 'T'), ChannelType(), optional=True),
                "cache_last_channel_next_len": NeuralType(tuple('B'), LengthsType(), optional=True),
            }
        )

    @property
    def output_types_for_export(self):
        """Returns definitions of module output ports."""
        return OrderedDict(
            {
                "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
                "cache_last_channel_next": NeuralType(('B', 'D', 'T', 'D'), ChannelType(), optional=True),
                "cache_last_time_next": NeuralType(('B', 'D', 'D', 'T'), ChannelType(), optional=True),
                "cache_last_channel_next_len": NeuralType(tuple('B'), LengthsType(), optional=True),
            }
        )

    @property
    def disabled_deployment_input_names(self):
        if not self.export_cache_support:
            return set(["cache_last_channel", "cache_last_time", "cache_last_channel_len"])
        else:
            return set()

    @property
    def disabled_deployment_output_names(self):
        if not self.export_cache_support:
            return set(["cache_last_channel_next", "cache_last_time_next", "cache_last_channel_next_len"])
        else:
            return set()

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
        d_visual_embeds: int = 1024,
        visual_downsampling_factor: int = 2, # by default, video is 25fps, audio feats are downsampled to 12.5fps.
        visual_conditioning_method: str = 'add', # add, film
        use_pre_pe_visual_conditioning: bool = True,
        use_visual_conditioning_on_all_layers: bool = False,
        visual_preprocessing_model: str = 'base', # base - downsample conv + LN, extra_conv  + LN + additional conv + LN
        conditioning_embed_aggr_method: str = 'avg',  # avg, wavg
        num_conditioning_embeds: int = 1, # 1 if only one model layer is used, otherwise # of layers. E.g. AV Hubert produces 25.
        modality_dropout_prob: float = 0.0,
        use_visual_adapter_encoder: bool = False,
        share_visual_preprocessing: bool = False,
        multi_speaker_visual_conditioning: bool = False,
        max_num_speakers: int = 8,
        use_stno: bool = True,
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

        self.d_visual_embeds = d_visual_embeds
        self.visual_downsampling_factor = visual_downsampling_factor
        self.visual_conditioning_method = visual_conditioning_method
        self.use_pre_pe_visual_conditioning = use_pre_pe_visual_conditioning
        self.use_visual_conditioning_on_all_layers = use_visual_conditioning_on_all_layers
        self.visual_preprocessing_model = visual_preprocessing_model
        self.conditioning_embed_aggr_method = conditioning_embed_aggr_method
        self.num_conditioning_embeds = num_conditioning_embeds
        self.modality_dropout_prob = modality_dropout_prob
        self.use_visual_adapter_encoder = use_visual_adapter_encoder
        self.share_visual_preprocessing = share_visual_preprocessing
        self.multi_speaker_visual_conditioning = multi_speaker_visual_conditioning
        self.max_num_speakers = max_num_speakers
        self.use_stno = use_stno

        if not self.use_stno:
            del self.fddts
            self.fddts = None

        if self.use_visual_adapter_encoder:
            self.visual_adapter_encoder = VisionAdapterEncoder(
                d_model=d_visual_embeds,
                num_layers=4,
                num_heads=n_heads,
                d_ff_ratio=2,
                dropout=dropout,
                dropout_pre_enc=dropout_pre_encoder,
            )
            assert self.num_conditioning_embeds == 1, "When using visual adapter encoder, num_conditioning_embeds must be 1."

        if self.share_visual_preprocessing:
            # Single shared preprocessing module for all uses
            self.shared_visual_processing = VisualProcessingModule(d_visual_embeds, d_model, visual_downsampling_factor, visual_preprocessing_model, conditioning_embed_aggr_method=conditioning_embed_aggr_method, num_conditioning_embeds=num_conditioning_embeds)

        self.VIS_CONDITIONING_MODULE_CLASS = MultiSpeakerVisualConditioningModule if self.multi_speaker_visual_conditioning else VisualConditioningModule

        if self.use_pre_pe_visual_conditioning:
            if not self.share_visual_preprocessing:
                # Create separate preprocessing module if not sharing
                self.pre_pe_visual_processing = VisualProcessingModule(d_visual_embeds, d_model, visual_downsampling_factor, visual_preprocessing_model, conditioning_embed_aggr_method=conditioning_embed_aggr_method, num_conditioning_embeds=num_conditioning_embeds)
            self.pre_pe_visual_conditioning = self.VIS_CONDITIONING_MODULE_CLASS(d_model, visual_conditioning_method, modality_dropout_prob=modality_dropout_prob, max_num_speakers=max_num_speakers)


        if self.use_visual_conditioning_on_all_layers:
            if not self.share_visual_preprocessing:
                # Per-layer preprocessing modules (only if not sharing)
                self.processing_modules = nn.ModuleList([
                    VisualProcessingModule(d_visual_embeds, d_model, visual_downsampling_factor, visual_preprocessing_model, conditioning_embed_aggr_method=conditioning_embed_aggr_method, num_conditioning_embeds=num_conditioning_embeds)
                    for _ in range(n_layers)
                ])
            self.conditioning_modules = nn.ModuleList([
                self.VIS_CONDITIONING_MODULE_CLASS(d_model, visual_conditioning_method, modality_dropout_prob=modality_dropout_prob, max_num_speakers=max_num_speakers)
                for _ in range(n_layers)
            ])
        else:
            raise NotImplementedError("Currently, at least one of use_pre_pe_visual_conditioning or use_visual_conditioning_on_all_layers must be True.")

    def unfreeze_visual_parameters(self):
        if self.share_visual_preprocessing:
            self.shared_visual_processing.train()
            for param in self.shared_visual_processing.parameters():
                param.requires_grad = True
        
        if self.use_pre_pe_visual_conditioning:
            if not self.share_visual_preprocessing:
                self.pre_pe_visual_processing.train()
                for param in self.pre_pe_visual_processing.parameters():
                    param.requires_grad = True
            
            self.pre_pe_visual_conditioning.train()
            for param in self.pre_pe_visual_conditioning.parameters():
                param.requires_grad = True

        if self.use_visual_conditioning_on_all_layers:
            if not self.share_visual_preprocessing:
                self.processing_modules.train()
                for param in self.processing_modules.parameters():
                    param.requires_grad = True
            
            self.conditioning_modules.train()
            for param in self.conditioning_modules.parameters():
                param.requires_grad = True

        if self.use_visual_adapter_encoder:
            self.visual_adapter_encoder.train()
            for param in self.visual_adapter_encoder.parameters():
                param.requires_grad = True

    @typecheck()
    def forward(
        self,
        audio_signal,
        length,
        cache_last_channel=None,
        cache_last_time=None,
        cache_last_channel_len=None,
        bypass_pre_encode=False,
        stno_mask=None,
        stno_mask_length=None,
        visual_embeds=None,
        visual_embed_lengths=None,
        num_speakers=None,
    ):
        """
        Forward function for the ConformerEncoderSTNO accepting an audio signal and its corresponding length.
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
            stno_mask=stno_mask,
            stno_mask_length=stno_mask_length,
            visual_embeds=visual_embeds,
            visual_embed_lengths=visual_embed_lengths,
            num_speakers=num_speakers,
        )

    def forward_internal(
        self,
        audio_signal,
        length,
        cache_last_channel=None,
        cache_last_time=None,
        cache_last_channel_len=None,
        bypass_pre_encode=False,
        stno_mask=None,
        stno_mask_length=None,
        visual_embeds=None,
        visual_embed_lengths=None,
        num_speakers=None,
    ):
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

        `stno_mask` and `stno_mask_length` are the speaker activity masks and their lengths.
        """
        if length is None:
            length = audio_signal.new_full(
                (audio_signal.size(0),), audio_signal.size(-1), dtype=torch.int64, device=audio_signal.device
            )

        if stno_mask is not None and stno_mask.numel() == 0 or not self.use_stno:
            stno_mask = None

        if not self.multi_speaker_visual_conditioning and len(visual_embeds.shape) == 5:
            # Squeeze the speaker dimension if not using multi-speaker visual conditioning
            visual_embeds = visual_embeds.squeeze(dim=2)  # (B, T, C, D)

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

        # Create the self-attention and padding masks
        pad_mask, att_mask = self._create_masks(
            att_context_size=cur_att_context_size,
            padding_length=padding_length,
            max_audio_length=max_audio_length,
            offset=offset,
            device=audio_signal.device,
        )

        if self.use_visual_adapter_encoder:
            vis_pad_mask, vis_att_mask = self._create_masks(att_context_size=[-1, -1], padding_length=visual_embed_lengths, max_audio_length=visual_embeds.size(1), offset=None, device=visual_embeds.device)
            visual_embeds = self.visual_adapter_encoder(visual_embeds.squeeze(dim=2), att_mask=vis_att_mask, pad_mask=vis_pad_mask).unsqueeze(dim=2)

        # Preprocess visual embeddings once if sharing preprocessing modules
        if self.share_visual_preprocessing:
            # (B, T, S, D)
            downsampled_visual_embeds_shared = self.shared_visual_processing(visual_embeds, audio_signal)

            if self.multi_speaker_visual_conditioning:
                # Zero out embeddings for non-present speakers.
                for i, n in enumerate(num_speakers):
                    downsampled_visual_embeds_shared[i, :, n:, :] = 0.0

        if self.use_pre_pe_visual_conditioning:
            if self.share_visual_preprocessing:
                downsampled_visual_embeds = downsampled_visual_embeds_shared
            else:
                downsampled_visual_embeds = self.pre_pe_visual_processing(visual_embeds, audio_signal)
            audio_signal = self.pre_pe_visual_conditioning(audio_signal=audio_signal, visual_embeds=downsampled_visual_embeds, att_mask=att_mask, num_speakers=num_speakers)

        # audio_signal = audio_signal + downsampled_visual_embeds
        audio_signal, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)

        # # Create the self-attention and padding masks
        # pad_mask, att_mask = self._create_masks(
        #     att_context_size=cur_att_context_size,
        #     padding_length=padding_length,
        #     max_audio_length=max_audio_length,
        #     offset=offset,
        #     device=audio_signal.device,
        # )

        if cache_last_channel is not None:
            pad_mask = pad_mask[:, cache_len:]
            if att_mask is not None:
                att_mask = att_mask[:, cache_len:]
            # Convert caches from the tensor to list
            cache_last_time_next = []
            cache_last_channel_next = []

        if stno_mask is not None:
            if stno_mask.shape[-1] != audio_signal.shape[1]:
                # print('DIFF SHAPES, stno_mask shape: ', stno_mask.shape, 'audio_signal shape: ', audio_signal.shape)
                if stno_mask.shape[-1] > audio_signal.shape[1]:
                    stno_mask = stno_mask[:, :, :audio_signal.shape[1]]
                else:
                    stno_mask = nn.functional.pad(stno_mask, (0, audio_signal.shape[1] - stno_mask.shape[-1]))
            
            assert stno_mask.shape[-1] == audio_signal.shape[1]

        for lth, (drop_prob, layer) in enumerate(zip(self.layer_drop_probs, self.layers)):
            original_signal = audio_signal
            if cache_last_channel is not None:
                cache_last_channel_cur = cache_last_channel[lth]
                cache_last_time_cur = cache_last_time[lth]
            else:
                cache_last_channel_cur = None
                cache_last_time_cur = None

            if stno_mask is not None:
                audio_signal = self.fddts[lth](audio_signal, stno_mask)

            if self.use_visual_conditioning_on_all_layers:
                if self.share_visual_preprocessing:
                    # Reuse the shared preprocessed visual embeddings
                    audio_signal = self.conditioning_modules[lth](audio_signal=audio_signal, visual_embeds=downsampled_visual_embeds, att_mask=att_mask, num_speakers=num_speakers)
                else:
                    # Use per-layer preprocessing
                    downsampled_visual_embeds = self.processing_modules[lth](visual_embeds, audio_signal)
                    audio_signal = self.conditioning_modules[lth](audio_signal=audio_signal, visual_embeds=downsampled_visual_embeds, att_mask=att_mask, num_speakers=num_speakers)
            else:
                raise NotImplementedError("Currently, at least one of use_pre_pe_visual_conditioning or use_visual_conditioning_on_all_layers must be True.")
            
            # if stno_mask is not None:
            #     audio_signal = self.fddts[lth](audio_signal, stno_mask)

            audio_signal = layer(
                x=audio_signal,
                att_mask=att_mask,
                pos_emb=pos_emb,
                pad_mask=pad_mask,
                cache_last_channel=cache_last_channel_cur,
                cache_last_time=cache_last_time_cur,
            )

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

        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal)

        # Reduction
        if self.reduction_position == -1:
            audio_signal, length = self.reduction_subsampling(x=audio_signal, lengths=length)

        audio_signal = torch.transpose(audio_signal, 1, 2)
        length = length.to(dtype=torch.int64)

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
            return audio_signal, length
