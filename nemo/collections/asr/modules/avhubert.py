import torch
import numpy as np
from torch import nn
from typing import Dict, List, Optional, Tuple, Any
from transformers import PreTrainedModel, Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Encoder, Wav2Vec2EncoderLayer, 
    is_deepspeed_zero3_enabled
)
from copy import deepcopy
from transformers.modeling_outputs import BaseModelOutput
from transformers.configuration_utils import PretrainedConfig


### RESNET ENCODER
import torch
import logging
import math
import torch.nn as nn
from collections import OrderedDict

import torch
from transformers.utils import ModelOutput
from typing import List, Optional, Union
from dataclasses import dataclass


logger = logging.getLogger(__name__)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def downsample_basic_block( inplanes, outplanes, stride ):
    return  nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes),
            )

def downsample_basic_block_v2( inplanes, outplanes, stride ):
    return  nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(outplanes),
            )



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, relu_type = 'relu' ):
        super(BasicBlock, self).__init__()

        assert relu_type in ['relu','prelu']

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        if relu_type == 'relu':
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif relu_type == 'prelu':
            self.relu1 = nn.PReLU(num_parameters=planes)
            self.relu2 = nn.PReLU(num_parameters=planes)
        else:
            raise Exception('relu type not implemented')

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, relu_type = 'relu', gamma_zero = False, avg_pool_downsample = False):
        self.inplanes = 64
        self.relu_type = relu_type
        self.gamma_zero = gamma_zero
        self.downsample_block = downsample_basic_block_v2 if avg_pool_downsample else downsample_basic_block

        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if self.gamma_zero:
            for m in self.modules():
                if isinstance(m, BasicBlock ):
                    m.bn2.weight.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):


        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = self.downsample_block( inplanes = self.inplanes, 
                                                 outplanes = planes * block.expansion, 
                                                 stride = stride )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, relu_type = self.relu_type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, relu_type = self.relu_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class ResEncoder(nn.Module):
    def __init__(self, relu_type, weights):
        super(ResEncoder, self).__init__()
        self.frontend_nout = 64
        self.backend_out = 512
        frontend_relu = nn.PReLU(num_parameters=self.frontend_nout) if relu_type == 'prelu' else nn.ReLU()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d( kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)
        if weights is not None:
            logger.info(f"Load {weights} for resnet")
            std = torch.load(weights, map_location=torch.device('cpu'))['model_state_dict']
            frontend_std, trunk_std = OrderedDict(), OrderedDict()
            for key, val in std.items():
                new_key = '.'.join(key.split('.')[1:])
                if 'frontend3D' in key:
                    frontend_std[new_key] = val
                if 'trunk' in key:
                    trunk_std[new_key] = val
            self.frontend3D.load_state_dict(frontend_std)
            self.trunk.load_state_dict(trunk_std)

    def forward(self, x):
        B, C, T, H, W = x.size()
        x = self.frontend3D(x)
        Tnew = x.shape[2]
        x = self.threeD_to_2D_tensor(x)
        x = self.trunk(x)
        x = x.view(B, Tnew, x.size(1))
        x = x.transpose(1, 2).contiguous()
        return x

    def threeD_to_2D_tensor(self, x):
        n_batch, n_channels, s_time, sx, sy = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.reshape(n_batch*s_time, n_channels, sx, sy)


# odim=args.odim,
# attention_dim=args.ddim,
# attention_heads=args.dheads,
# linear_units=args.dunits,
# num_blocks=args.dlayers,
# dropout_rate=args.dropout_rate,
# positional_dropout_rate=args.dropout_rate,
# self_attention_dropout_rate=args.transformer_attn_dropout_rate,
# src_attention_dropout_rate=args.transformer_attn_dropout_rate,

class AVHubertAVSRConfig(PretrainedConfig):
    model_type = "avhubert_avsr"

    def __init__(
        self,
        odim=5049,
        adim=1024,
        aheads=12,
        eunits=3072,
        elayers=12,
        transformer_input_layer="conv3d",
        dropout_rate=0.1,
        transformer_attn_dropout_rate=0.1,
        transformer_encoder_attn_layer_type="rel_mha",
        macaron_style=True,
        use_cnn_module=True,
        cnn_module_kernel=31,
        zero_triu=False,
        a_upsample_ratio=1,
        relu_type="swish",
        ddim=1024,
        dheads=16,
        dunits=3072,
        dlayers=6,
        lsm_weight=0.1,
        transformer_length_normalized_loss=False,
        mtlalpha=0.1,
        ctc_type="builtin",
        rel_pos_type="latest",
        # aux_adim=768,
        # aux_aheads=12,
        # aux_eunits=3072,
        # aux_elayers=12,
        # aux_transformer_input_layer="conv1d",
        # aux_dropout_rate=0.1,
        # aux_transformer_attn_dropout_rate=0.1,
        # aux_transformer_encoder_attn_layer_type="rel_mha",
        # aux_macaron_style=True,
        # aux_use_cnn_module=True,
        # aux_cnn_module_kernel=31,
        # aux_zero_triu=False,
        # aux_a_upsample_ratio=1,
        # aux_relu_type="swish",
        # aux_dunits=3072,
        # aux_dlayers=6,
        # aux_lsm_weight=0.1,
        # aux_transformer_length_normalized_loss=False,
        # aux_mtlalpha=0.1,
        # aux_ctc_type="builtin",
        # aux_rel_pos_type="latest",
        fusion_hdim=8192,
        fusion_norm="batchnorm",
        
        hidden_size=1024,
        num_attention_heads=16,
        activation_dropout = 0.0,
        activation_function = "relu",
        adapter_attn_dim = None,
        adapter_kernel_size = 3,
        adapter_stride = 2,
        add_adapter = False,
        apply_spec_augment = True,
        attention_dropout = 0.1,
        audio_dropout = 0.5,
        audio_feat_dim = 104,
        bos_token_id = 1,
        classifier_proj_size = 256,
        codevector_dim = 256,
        contrastive_logits_temperature = 0.1,
        conv_bias = False,
        conv_channels = 1024,
        conv_dim = [512,512,512,512,512,512,512],
        conv_kernel = [10,3,3,3,3,2,2],
        conv_kernel_sizes = [5,5],
        conv_stride = [5,2,2,2,2,2,2],
        ctc_loss_reduction = "sum",
        ctc_zero_infinity = False,
        d_model = 1024,
        decoder_attention_heads = 8,
        decoder_ffn_dim = 4096,
        decoder_layerdrop = 0.0,
        decoder_layers = 9,
        decoder_start_token_id = 2,
        diversity_loss_weight = 0.1,
        do_stable_layer_norm = False,
        dropout = 0.1,
        dropout_features = 0.1,
        dropout_input = 0.1,
        encoder_attention_heads = 16,
        encoder_embed_dim = 1024,
        encoder_ffn_dim = 2048,
        encoder_layerdrop = 0.0,
        encoder_layers = 12,
        eos_token_id = 2,
        feat_extract_activation = "gelu",
        feat_extract_norm = "group",
        feat_proj_dropout = 0.1,
        feat_quantizer_dropout = 0.0,
        feature_grad_mult = 0.1,
        final_dim = 256,
        final_dropout = 0.0,
        freeze_feat_extract_train = True,
        hidden_act = "gelu",
        hidden_dropout = 0.1,
        init_std = 0.02,
        initializer_range = 0.02,
        input_channels = 1,
        input_feat_per_channel = 80,
        intermediate_size = 4096,
        is_encoder_decoder = True,
        label_rate = 25,
        layer_norm_eps = 1e-05,
        layerdrop = 0.0,
        logit_temp = 0.1,
        mask_channel_length = 10,
        mask_channel_min_space = 1,
        mask_channel_other = 0.0,
        mask_channel_prob = 0.0,
        mask_channel_selection = "static",
        mask_feature_length = 10,
        mask_feature_min_masks = 0,
        mask_feature_prob = 0.0,
        mask_length_audio = 10,
        mask_length_image = 5,
        mask_min_space = 1,
        mask_other = 0.0,
        mask_prob_audio = 0.8,
        mask_prob_image = 0.3,
        mask_selection = "static",
        mask_time_length = 10,
        mask_time_min_masks = 2,
        mask_time_min_space = 1,
        mask_time_other = 0.0,
        mask_time_prob = 0.0,
        mask_time_selection = "static",
        masking_type = "input",
        max_source_positions = 6000,
        max_target_positions = 2048,
        modality_dropout = 0.5,
        modality_fuse = "concat",
        modality= "av",
        model_type = "speech_to_text",
        no_mask_channel_overlap = False,
        no_mask_overlap = False,
        no_mask_time_overlap = False,
        num_adapter_layers = 3,
        num_classes = 2004,
        num_codevector_groups = 2,
        num_codevectors_per_group = 320,
        num_conv_layers = 2,
        num_conv_pos_embedding_groups = 16,
        num_conv_pos_embeddings = 128,
        num_dictionaries = 1,
        num_feat_extract_layers = 7,
        num_hidden_layers = 24,
        num_negatives = 100,
        output_hidden_size = 1024,
        pad_token_id = 1,
        proj_codevector_dim = 256,
        resnet_relu_type = "prelu",
        resnet_weights = None,
        sample_rate = 25,
        scale_embedding = None,
        selection_type = "same_seq",
        sim_type = "cosine",
        skip_masked = False,
        skip_nomask = False,
        sub_encoder_layers = 0,
        target_glu = False,
        tdnn_dilation = [1,2,3,1,1],
        tdnn_dim = [512,512,512,512,1500],
        tdnn_kernel = [5,3,3,1,1],
        untie_final_proj = True,
        use_cache = True,
        use_weighted_layer_sum = False,
        vocab_size = 1000,
        xvector_output_dim = 512,
        
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.odim = odim
        self.adim = adim
        self.aheads = aheads
        self.eunits = eunits
        self.elayers = elayers
        self.transformer_input_layer = transformer_input_layer
        self.dropout_rate = dropout_rate
        self.transformer_attn_dropout_rate = transformer_attn_dropout_rate
        self.transformer_encoder_attn_layer_type = transformer_encoder_attn_layer_type
        self.macaron_style = macaron_style
        self.use_cnn_module = use_cnn_module
        self.cnn_module_kernel = cnn_module_kernel
        self.zero_triu = zero_triu
        self.a_upsample_ratio = a_upsample_ratio
        self.relu_type = relu_type
        self.ddim = ddim
        self.dheads = dheads
        self.dunits = dunits
        self.dlayers = dlayers
        self.lsm_weight = lsm_weight
        self.transformer_length_normalized_loss = transformer_length_normalized_loss
        self.mtlalpha = mtlalpha
        self.ctc_type = ctc_type
        self.rel_pos_type = rel_pos_type
        # self.aux_adim = aux_adim
        # self.aux_aheads = aux_aheads
        # self.aux_eunits = aux_eunits
        # self.aux_elayers = aux_elayers
        # self.aux_transformer_input_layer = aux_transformer_input_layer
        # self.aux_dropout_rate = aux_dropout_rate
        # self.aux_transformer_attn_dropout_rate = aux_transformer_attn_dropout_rate
        # self.aux_transformer_encoder_attn_layer_type = aux_transformer_encoder_attn_layer_type
        # self.aux_macaron_style = aux_macaron_style
        # self.aux_use_cnn_module = aux_use_cnn_module
        # self.aux_cnn_module_kernel = aux_cnn_module_kernel
        # self.aux_zero_triu = aux_zero_triu
        # self.aux_a_upsample_ratio = aux_a_upsample_ratio
        # self.aux_relu_type = aux_relu_type
        # self.aux_dunits = aux_dunits
        # self.aux_dlayers = aux_dlayers
        # self.aux_lsm_weight = aux_lsm_weight
        # self.aux_transformer_length_normalized_loss = aux_transformer_length_normalized_loss
        # self.aux_mtlalpha = aux_mtlalpha
        # self.aux_ctc_type = aux_ctc_type
        # self.aux_rel_pos_type = aux_rel_pos_type
        self.fusion_hdim = fusion_hdim
        self.fusion_norm = fusion_norm
        
        
        
        self.hidden_size = hidden_size
        self.num_attention_heads= num_attention_heads
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.adapter_attn_dim = adapter_attn_dim
        self.adapter_kernel_size = adapter_kernel_size
        self.adapter_stride = adapter_stride
        self.add_adapter = add_adapter
        self.apply_spec_augment = apply_spec_augment
        self.attention_dropout = attention_dropout
        self.audio_dropout = audio_dropout
        self.audio_feat_dim = audio_feat_dim
        self.bos_token_id = bos_token_id
        self.classifier_proj_size = classifier_proj_size
        self.codevector_dim = codevector_dim
        self.contrastive_logits_temperature = contrastive_logits_temperature
        self.conv_bias = conv_bias
        self.conv_channels = conv_channels
        self.conv_dim = conv_dim
        self.conv_kernel = conv_kernel
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_stride = conv_stride
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity
        self.d_model = d_model
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layerdrop = decoder_layerdrop
        self.decoder_layers = decoder_layers
        self.decoder_start_token_id = decoder_start_token_id
        self.diversity_loss_weight = diversity_loss_weight
        self.do_stable_layer_norm = do_stable_layer_norm
        self.dropout = dropout
        self.dropout_features = dropout_features
        self.dropout_input = dropout_input
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layerdrop = encoder_layerdrop
        self.encoder_layers = encoder_layers
        self.eos_token_id = eos_token_id
        self.feat_extract_activation = feat_extract_activation
        self.feat_extract_norm = feat_extract_norm
        self.feat_proj_dropout = feat_proj_dropout
        self.feat_quantizer_dropout = feat_quantizer_dropout
        self.feature_grad_mult = feature_grad_mult
        self.final_dim = final_dim
        self.final_dropout = final_dropout
        self.freeze_feat_extract_train = freeze_feat_extract_train
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.init_std = init_std
        self.initializer_range = initializer_range
        self.input_channels = input_channels
        self.input_feat_per_channel = input_feat_per_channel
        self.intermediate_size = intermediate_size
        self.is_encoder_decoder = is_encoder_decoder
        self.label_rate = label_rate
        self.layer_norm_eps = layer_norm_eps
        self.layerdrop = layerdrop
        self.logit_temp = logit_temp
        self.mask_channel_length = mask_channel_length
        self.mask_channel_min_space = mask_channel_min_space
        self.mask_channel_other = mask_channel_other
        self.mask_channel_prob = mask_channel_prob
        self.mask_channel_selection = mask_channel_selection
        self.mask_feature_length = mask_feature_length
        self.mask_feature_min_masks = mask_feature_min_masks
        self.mask_feature_prob = mask_feature_prob
        self.mask_length_audio = mask_length_audio
        self.mask_length_image = mask_length_image
        self.mask_min_space = mask_min_space
        self.mask_other = mask_other
        self.mask_prob_audio = mask_prob_audio
        self.mask_prob_image = mask_prob_image
        self.mask_selection = mask_selection
        self.mask_time_length = mask_time_length
        self.mask_time_min_masks = mask_time_min_masks
        self.mask_time_min_space = mask_time_min_space
        self.mask_time_other = mask_time_other
        self.mask_time_prob = mask_time_prob
        self.mask_time_selection = mask_time_selection
        self.masking_type = masking_type
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.modality_dropout = modality_dropout
        self.modality_fuse = modality_fuse
        self.modality = modality
        self.model_type = model_type
        self.no_mask_channel_overlap = no_mask_channel_overlap
        self.no_mask_overlap = no_mask_overlap
        self.no_mask_time_overlap = no_mask_time_overlap
        self.num_adapter_layers = num_adapter_layers
        self.num_classes = num_classes
        self.num_codevector_groups = num_codevector_groups
        self.num_codevectors_per_group = num_codevectors_per_group
        self.num_conv_layers = num_conv_layers
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_dictionaries = num_dictionaries
        self.num_feat_extract_layers = num_feat_extract_layers
        self.num_hidden_layers = num_hidden_layers
        self.num_negatives = num_negatives
        self.output_hidden_size = output_hidden_size
        self.pad_token_id = pad_token_id
        self.proj_codevector_dim = proj_codevector_dim
        self.resnet_relu_type = resnet_relu_type
        self.resnet_weights = resnet_weights
        self.sample_rate = sample_rate
        self.scale_embedding = scale_embedding
        self.selection_type = selection_type
        self.sim_type = sim_type
        self.skip_masked = skip_masked
        self.skip_nomask = skip_nomask
        self.sub_encoder_layers = sub_encoder_layers
        self.target_glu = target_glu
        self.tdnn_dilation = tdnn_dilation
        self.tdnn_dim = tdnn_dim
        self.tdnn_kernel = tdnn_kernel
        self.untie_final_proj = untie_final_proj
        self.use_cache = use_cache
        self.use_weighted_layer_sum = use_weighted_layer_sum
        self.vocab_size = vocab_size
        self.xvector_output_dim = xvector_output_dim

def find_runs(x):
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths


def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape
    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    batch_indexes, starts, ends = [], [], []
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True
        vals, run_starts, run_lengths = find_runs(mask[i])
        start_indices, lengths = run_starts[vals == True], run_lengths[vals == True]
        starts.append(start_indices)
        ends.append(start_indices+lengths)
        batch_indexes.append(np.zeros([len(start_indices)])+i)
    return mask, np.concatenate(starts).astype(np.int64), np.concatenate(ends).astype(np.int64), np.concatenate(batch_indexes).astype(np.int64)

class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None

def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)

class SubModel(nn.Module):
    def __init__(self, resnet=None, input_dim=None, cfg=None):
        super().__init__()
        self.resnet = resnet
        self.proj = nn.Linear(input_dim, cfg.encoder_embed_dim)

    def forward(self, x):
        if self.resnet is not None:
            x = self.resnet(x)
        x = self.proj(x.transpose(1, 2))
        x = x.transpose(1, 2)
        return x

@dataclass
class AVHubertAVSROutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_ctc: Optional[torch.FloatTensor] = None
    loss_att: Optional[torch.FloatTensor] = None
    acc: Optional[torch.FloatTensor] = None

class AVHubertAVSR(PreTrainedModel):
    config_class = AVHubertAVSRConfig
    
    def __init__(self, config: AVHubertAVSRConfig):
        super().__init__(config)
        self.avsr = E2E(config)
    
    def forward(self, 
        videos, 
        audios, 
        labels,
        video_lengths, 
        audio_lengths, 
        label_lengths
    ):
        loss, loss_ctc, loss_att, acc = self.avsr(videos, audios, video_lengths, audio_lengths, labels)
        return AVHubertAVSROutput(
            loss=loss,
            loss_ctc=loss_ctc,
            loss_att=loss_att,
            acc=acc
        )


class AVHubertModel(PreTrainedModel):
    config_class = Wav2Vec2Config
    base_model_prefix = "avhubert"
    # main_input_name = "input_values"
    # supports_gradient_checkpointing = True
    # _supports_flash_attn_2 = True
    # _supports_sdpa = True
    
    def __init__(
        self,
        cfg: Wav2Vec2Config,
    ) -> None:
        super().__init__(cfg)
        # logger.info(f"HubertModel Config: {cfg}")

        feature_ds_rate = 1
        self.feat2tar_ratio = cfg.label_rate * feature_ds_rate / cfg.sample_rate
        sub_cfg = deepcopy(cfg)
        sub_cfg.encoder_layers = sub_cfg.sub_encoder_layers
        resnet = ResEncoder(relu_type=cfg.resnet_relu_type, weights=cfg.resnet_weights)
        self.feature_extractor_audio = SubModel(resnet=None, input_dim=cfg.audio_feat_dim, cfg=sub_cfg)
        self.feature_extractor_video = SubModel(resnet=resnet, input_dim=resnet.backend_out, cfg=sub_cfg)
        self.modality_dropout, self.audio_dropout = cfg.modality_dropout, cfg.audio_dropout
        self.modality_fuse = cfg.modality_fuse
        self.encoder_embed_dim = cfg.encoder_embed_dim
        if self.modality_fuse == 'concat':
            self.embed = cfg.encoder_embed_dim * 2
        elif self.modality_fuse == 'add':
            self.embed = cfg.encoder_embed_dim
        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.mask_prob_image, self.mask_prob_audio = cfg.mask_prob_image, cfg.mask_prob_audio
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length_image, self.mask_length_audio = cfg.mask_length_image, cfg.mask_length_audio
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult
        self.logit_temp = cfg.logit_temp
        self.skip_masked = cfg.skip_masked
        self.skip_nomask = cfg.skip_nomask
        self.sim_type = cfg.sim_type
        self.selection_type = cfg.selection_type
        self.masking_type = cfg.masking_type
        self.modality = cfg.modality

        final_dim = (
            cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim
        )

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.audio_feat_dim).uniform_() if self.masking_type == 'input' else torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = AVHubertEncoder(cfg)
        
        self.layer_norm = LayerNorm(self.embed)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.untie_final_proj = cfg.untie_final_proj
        # if self.untie_final_proj:
        #     self.final_proj = nn.Linear(
        #         cfg.encoder_embed_dim, final_dim * cfg.num_dictionaries
        #     )
        # else:
        #     self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)

        self.num_classes = [cfg.num_classes]
        self.label_embs_concat = nn.Parameter(
            torch.FloatTensor(sum(self.num_classes), final_dim)
        )
        nn.init.uniform_(self.label_embs_concat)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def apply_input_mask(self, x, padding_mask, target_list):
        B, C, T = x.shape[:3]
        is_audio = True if len(x.shape) == 3 else False
        if is_audio:
            mask_prob, mask_length = self.mask_prob_audio, self.mask_length_audio
        else:
            mask_prob, mask_length = self.mask_prob_image, self.mask_length_image
        if mask_prob > 0:

            mask_indices, starts, ends, batch_indexes = compute_mask_indices(
                (B, T),
                padding_mask,
                mask_prob,
                mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices_np = mask_indices
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = x.transpose(1, 2).contiguous() # [B, T, C, H, W]
            if B == 1:
                x[mask_indices] = 0
            elif is_audio:
                x[mask_indices] = self.mask_emb
            elif self.selection_type == 'same_other_seq':
                perm = (torch.arange(B) + torch.randint(low=1, high=B, size=(1,))) % B
                x_perm = x[perm]
                x[mask_indices] = x_perm[mask_indices]
            elif self.selection_type == 'same_seq':
                batch_indexes_, other_indexes = [], []
                for batch_index, start, end in zip(batch_indexes, starts, ends):
                    length = end-start
                    other_start = np.setdiff1d(np.arange(T), np.arange(max(0, start-length), end))
                    if len(other_start) > 0:
                        other_start = np.random.choice(other_start, size=1)
                    else:
                        other_start = 0
                    other_end = other_start + length
                    other_indexes.append(np.arange(other_start, other_end).clip(max=T-1))
                    batch_indexes_.append(np.zeros([length], dtype=np.int64)+batch_index)
                batch_indexes, other_indexes = np.concatenate(batch_indexes_), np.concatenate(other_indexes)
                x[mask_indices] = x[batch_indexes, other_indexes]

            x = x.transpose(1, 2).contiguous()
        else:
            mask_indices = None

        # if self.mask_channel_prob > 0:
        #     logger.info(f"No mask channel prob for input masking")
        return x, mask_indices

    def apply_feature_mask(self, x, padding_mask, target_list):
        B, T, C = x.shape
        assert self.mask_prob_audio == self.mask_prob_image and self.mask_length_audio == self.mask_length_image, f"masking prob/length for image/audio be same for feature masking"
        mask_prob, mask_length = self.mask_prob_audio, self.mask_length_image
        if mask_prob > 0:
            mask_indices, _, _, _ = compute_mask_indices(
                (B, T),
                padding_mask,
                mask_prob,
                mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices, _, _, _ = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def forward_features(self, source: torch.Tensor, modality: str) -> torch.Tensor:
        extractor = eval(f"self.feature_extractor_{modality}")
        if self.feature_grad_mult > 0:
            features = extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = extractor(source)
        return features

    def forward_targets(
            self, features: torch.Tensor, mask_indices: torch.Tensor, target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels
        feat_tsz = features.size(2)
        targ_tsz = min([t.size(1) for t in target_list])
        if self.feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / self.feat2tar_ratio)
            features = features[..., :feat_tsz]
            if mask_indices is not None:
                mask_indices = mask_indices[..., :feat_tsz]
        target_inds = torch.arange(feat_tsz).float() * self.feat2tar_ratio
        target_list = [t[:, target_inds.long()] for t in target_list]
        return features, mask_indices, target_list

    def forward_padding_mask(
        self, features: torch.Tensor, padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(
            padding_mask.size(0), features.size(1), -1
        )
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def compute_logits(self, feats, emb_mat):
        # feats: [B, T, F], emb_mat: [V, F]
        if self.sim_type == 'dot':
            logits = torch.matmul(feats, emb_mat.transpose(0, 1))
        elif self.sim_type == 'cosine':
            batch_size, timesteps, emb_dim = feats.size()
            feats_ = feats.view(-1, emb_dim)
            nom = (feats_.unsqueeze(dim=1) * emb_mat.unsqueeze(dim=0)).sum(dim=-1) # [B*T, V]
            denom = (feats_**2).sum(dim=-1).sqrt().unsqueeze(dim=1) * (emb_mat**2).sum(dim=-1).sqrt().unsqueeze(dim=0) # [B*T, V]
            logits = (nom/denom.clamp(min=1e-6)).view(batch_size, timesteps, -1)
        else:
            raise NotImplementedError
        logits = logits / self.logit_temp
        return logits

    def forward_gen(
        self,
        source: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
        video: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        src_audio, src_video = source['audio'], source['video']
        if mask and self.masking_type == 'input':
            src_video, mask_indices_video = self.apply_input_mask(src_video, padding_mask, target_list)
            src_audio, mask_indices_audio = self.apply_input_mask(src_audio, padding_mask, target_list)
            mask_indices = torch.logical_or(mask_indices_audio, mask_indices_video)
        else:
            src_audio, src_video, mask_indices = src_audio, src_video, None
        
        if src_video is not None:
            features_video = self.forward_features(src_video, modality='video')
        else:
            features_video = torch.zeros_like(features_audio)

        if src_audio is not None:
            features_audio = self.forward_features(src_audio, modality='audio') # features: [B, F, T]
        else:
            features_audio = torch.zeros_like(features_video)
        
        # if self.modality == 'audio':
        #     features_video = 0 * features_video
        # elif self.modality == 'video':
        #     features_audio = 0 * features_audio
        # else:
        #     if self.training:
        #         modality_drop_prob, audio_drop_prob = np.random.random(), np.random.random()
        #         if modality_drop_prob < self.modality_dropout:
        #             if audio_drop_prob < self.audio_dropout:
        #                 features_audio = 0 * features_audio
        #             else:
        #                 features_video = 0 * features_video
        
        if self.modality_fuse == 'concat':
            features = torch.cat([features_audio, features_video], dim=1)
        elif self.modality_fuse == 'add':
            features = features_audio + features_video
        if target_list is not None:
            features, mask_indices, target_list = self.forward_targets(features, mask_indices, target_list)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        if self.masking_type == 'feature' and mask:
            x, mask_indices = self.apply_feature_mask(features, padding_mask, target_list)
        else:
            x = features

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        
        x = self.encoder(x, attention_mask=padding_mask, output_hidden_states=True)
        # x = self.encoder(
        #     x,
        #     # attention_mask=padding_mask,
        #     # layer=None if output_layer is None else output_layer - 1
        # )[0]

        if features_only:
            return {"x": x, "padding_mask": padding_mask, "features": features}

        label_embs_list = self.label_embs_concat.split(self.num_classes, 0)
        proj_x = self.final_proj(x)
        if self.untie_final_proj:
            proj_x_list = proj_x.chunk(len(self.num_classes), dim=-1)
        else:
            proj_x_list = [proj_x for _ in self.num_classes]
        logit_list = [self.compute_logits(proj, emb).view(-1, num_class) for proj, emb, num_class in zip(proj_x_list, label_embs_list, self.num_classes)] # [[B*T, V]]
        mask, unmask = torch.logical_and(mask_indices, ~padding_mask).view(-1), torch.logical_and(~mask_indices, ~padding_mask).view(-1) # [B*T]
        logit_m_list, logit_u_list = [logit[mask] for logit in logit_list], [logit[unmask] for logit in logit_list]
        target_m_list, target_u_list = [target.view(-1)[mask].long() for target in target_list], [target.view(-1)[unmask].long() for target in target_list]
        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "target_m_list": target_m_list,
            "target_u_list": target_u_list,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }
        return result

    def forward(
        self,
        input_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        video: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.forward_gen(
            {"audio": input_features, "video": video},
            padding_mask=attention_mask,
            mask=False,
            features_only=True,
            output_layer=None,
        )
        return res["x"]
        # feature = res["x"]
        # return BaseModelOutput(last_hidden_state=feature, hidden_states=None, attentions=None)

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.forward_gen(
            source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )
        feature = res["features"] if ret_conv else res["x"]
        return feature, res["padding_mask"]

    def extract_finetune(self, source, padding_mask=None, mask=False, ret_conv=False, output_layer=None):
        src_audio, src_video = source['audio'], source['video']
        if mask and self.masking_type == 'input':
            src_video, mask_indices_video = self.apply_input_mask(src_video, padding_mask, target_list=None)
            src_audio, mask_indices_audio = self.apply_input_mask(src_audio, padding_mask, target_list=None)
            mask_indices = torch.logical_or(mask_indices_audio, mask_indices_video) # mask_indices not used in fine-tuning
        else:
            src_audio, src_video, mask_indices = src_audio, src_video, None

        if src_audio is not None and src_video is None:
            features_audio = self.forward_features(src_audio, modality='audio') # features: [B, F, T]
            features_video = features_audio.new_zeros(features_audio.size(0), self.encoder_embed_dim, features_audio.size(-1))
        elif src_audio is None and src_video is not None:
            features_video = self.forward_features(src_video, modality='video')
            features_audio = features_video.new_zeros(features_video.size(0), self.encoder_embed_dim, features_video.size(-1))
        elif src_audio is not None and src_video is not None:
            features_video = self.forward_features(src_video, modality='video')
            features_audio = self.forward_features(src_audio, modality='audio') # features: [B, F, T]

        if self.modality_fuse == 'concat':
            features = torch.cat([features_audio, features_video], dim=1)
        elif self.modality_fuse == 'add':
            features = features_audio + features_video
        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)
        x = features
        mask_indices = None

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x = self.encoder(
            x,
            # padding_mask=padding_mask,
            # layer=None if output_layer is None else output_layer - 1
        )[0]

        return x, padding_mask


    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []
        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")

        return extra_losses, names

    def remove_pretraining_modules(self):
        self.target_glu = None
        self.final_proj = None

    def get_logits(self, net_output, is_masked=True):
        raise NotImplementedError

    def get_targets(self, net_output, is_masked=True):
        raise NotImplementedError

    def compute_nce(self, x, pos, negs):
        neg_is_pos = (pos == negs).all(-1)
        pos = pos.unsqueeze(0)
        targets = torch.cat([pos, negs], dim=0)

        logits = torch.cosine_similarity(
            x.float(), targets.float(), dim=-1
        ).type_as(x)
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
        return logits

class AVHubertEncoder(Wav2Vec2Encoder):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([AVHubertEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self._use_flash_attention_2 = False
    def forward(
        self,
        hidden_states: torch.tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0
            if self._use_flash_attention_2:
                # 2d mask is passed through the layers
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            else:
                # extend attention_mask
                attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
                attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
                attention_mask = attention_mask.expand(
                    attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
                )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        # hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                        output_attentions,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
        
class AVHubertEncoderLayer(Wav2Vec2EncoderLayer):
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        # hidden_states = self.layer_norm(hidden_states)
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = residual + self.feed_forward(hidden_states)
        # hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class E2E(torch.nn.Module):
    def __init__(self, args, ignore_id=-1):
        torch.nn.Module.__init__(self)
        self.encoder = AVHubertModel(args)

        self.proj_decoder = None
        if args.adim != args.ddim:
            self.proj_decoder = torch.nn.Linear(args.adim, args.ddim)

        # if args.mtlalpha < 1:
        #     self.decoder = Decoder(
        #         odim=args.odim,
        #         attention_dim=args.ddim,
        #         attention_heads=args.dheads,
        #         linear_units=args.dunits,
        #         num_blocks=args.dlayers,
        #         dropout_rate=args.dropout_rate,
        #         positional_dropout_rate=args.dropout_rate,
        #         self_attention_dropout_rate=args.transformer_attn_dropout_rate,
        #         src_attention_dropout_rate=args.transformer_attn_dropout_rate,
        #     )
        # else:
        #     self.decoder = None
        self.blank = 0
        self.sos = args.odim - 1
        self.eos = args.odim - 1
        self.odim = args.odim
        self.ignore_id = ignore_id


    def forward(self, video, audio, video_lengths, audio_lengths, label):
        return None, None, None
        # video_padding_mask = make_non_pad_mask(video_lengths).to(video.device)
        # # attention_mask = 
        # avhubert_features = self.encoder(
        #     input_features = audio, 
        #     video = video,
        #     attention_mask = video_padding_mask
        # )
        
        # # avhubert_features = self.encoder(
        # #     input_features = audio, 
        # #     video = video,
        # #     attention_mask = None
        # # )
        #     # attention_mask: Optional[torch.Tensor] = None,

        # # audio_lengths = torch.div(audio_lengths, 640, rounding_mode="trunc")
        # # audio_padding_mask = make_non_pad_mask(audio_lengths).to(video.device).unsqueeze(-2)

        # # audio_feat, _ = self.aux_encoder(audio, audio_padding_mask)

        # # x = self.fusion(torch.cat((video_feat, audio_feat), dim=-1))
        # x = avhubert_features.last_hidden_state
        # # ctc loss
        # loss_ctc, ys_hat = self.ctc(x, video_lengths, label)

        # if self.proj_decoder:
        #     x = self.proj_decoder(x)

        # # decoder loss
        # ys_in_pad, ys_out_pad = add_sos_eos(label, self.sos, self.eos, self.ignore_id)
        # ys_mask = target_mask(ys_in_pad, self.ignore_id)
        # pred_pad, _ = self.decoder(ys_in_pad, ys_mask, x, video_padding_mask.unsqueeze(-2))
        # loss_att = self.criterion(pred_pad, ys_out_pad)
        # loss = self.mtlalpha * loss_ctc + (1 - self.mtlalpha) * loss_att

        # acc = th_accuracy(
        #     pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
        # )

        # return loss, loss_ctc, loss_att, acc


def make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 1],
                 [0, 0, 0, 1]],
                [[0, 0, 1, 1],
                 [0, 0, 1, 1]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_pad_mask(lengths, xs, 1)
        tensor([[[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]]], dtype=torch.uint8)
        >>> make_pad_mask(lengths, xs, 2)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if maxlen is None:
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)
    else:
        assert xs is None
        assert maxlen >= int(max(lengths))

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


def make_non_pad_mask(lengths, xs=None, length_dim=-1):
    """Make mask tensor containing indices of non-padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        ByteTensor: mask tensor containing indices of padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1],
                 [1, 1, 1, 1]],
                [[1, 1, 1, 0],
                 [1, 1, 1, 0]],
                [[1, 1, 0, 0],
                 [1, 1, 0, 0]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_non_pad_mask(lengths, xs, 1)
        tensor([[[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]]], dtype=torch.uint8)
        >>> make_non_pad_mask(lengths, xs, 2)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=torch.uint8)

    """
    return ~make_pad_mask(lengths, xs, length_dim)