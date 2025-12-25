import torch
import torch.nn as nn
from transformers import TorchAoConfig, AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
import torchvision
from nemo.collections.asr.parts.submodules.multi_head_attention import RelPositionalEncoding
from nemo.collections.asr.parts.submodules.conformer_modules import ConformerLayer
from transformers.modeling_outputs import BaseModelOutput

class DINOv3VRSEncoder(torch.nn.Module):
    def __init__(self, model_name: str = "dinov3-vith16plus-pretrain-lvd1689m", 
                 freeze_dino: bool = True, 
                 dino_batch_size: int = 16,
                 use_register_tokens: bool = False,
                 conformer_n_layers: int = 12,
                 conformer_expansion_factor: int = 4,
                 conformr_n_heads: int = 8,
                 conformer_conv_kernel_size: int = 9,
                 conformer_dropout: float = 0.1,
                 conformer_use_bias: bool = False,):
        super().__init__()
        self.dino_model = AutoModel.from_pretrained(
            f"facebook/{model_name}",
            # dtype=torch.bfloat16,
            # device_map=self.device,
        )
        self.freeze_dino = freeze_dino
        self.dino_batch_size = dino_batch_size
        self.use_register_tokens = use_register_tokens
        self.dino_embed_dim = self.dino_model.config.hidden_size

        self.num_patches = (self.dino_model.config.image_size // self.dino_model.config.patch_size) ** 2
        if self.use_register_tokens:
            self.num_patches += self.dino_model.config.num_register_tokens

        if self.freeze_dino:
            self.dino_model.eval()
            for param in self.dino_model.parameters():
                param.requires_grad = False

        self.inference_ctx_fn = torch.no_grad if self.freeze_dino else torch.enable_grad

        self.attn_pooling_query = torch.nn.Parameter(
            torch.randn(1, 1, self.dino_embed_dim)
        )
        self.attn_pooling = torch.nn.MultiheadAttention(
            embed_dim=self.dino_embed_dim,
            num_heads=4,
            batch_first=True,
        )

        self.conv_preproccessing = nn.Conv1d(self.dino_embed_dim, self.dino_embed_dim, kernel_size=9, padding=4, bias=False)

        self.conformer_n_layers = conformer_n_layers
        self.conformer_expansion_factor = conformer_expansion_factor
        self.conformer_n_heads = conformr_n_heads
        self.conformer_kernel_size = conformer_conv_kernel_size
        self.conformer_dropout = conformer_dropout
        self.conformer_use_bias = conformer_use_bias
        self.self_attention_model = 'rel_pos'
        self.att_context_style = 'regular'
        self.attn_context_size = [-1, -1]
        self.conformer_layers = nn.ModuleList()
        for _ in range(self.conformer_n_layers):
            layer = ConformerLayer(
                d_model=self.dino_embed_dim,
                d_ff=self.dino_embed_dim * self.conformer_expansion_factor,
                self_attention_model=self.self_attention_model,
                global_tokens=0,
                global_tokens_spacing=1,
                global_attn_separate=1,
                n_heads=self.conformer_n_heads,
                conv_kernel_size=self.conformer_kernel_size,
                conv_norm_type='batch_norm',
                conv_context_size=None,
                dropout=self.conformer_dropout,
                dropout_att=0.0,
                pos_bias_u=None,
                pos_bias_v=None,
                att_context_size=self.attn_context_size,
                use_bias=self.conformer_use_bias,
                use_pytorch_sdpa=False,
                use_pytorch_sdpa_backends=None,
            )
            self.conformer_layers.append(layer)

        max_pos = 45000
        self.pos_enc = RelPositionalEncoding(
            d_model=self.dino_embed_dim,
            dropout_rate=self.conformer_dropout,
            max_len=max_pos,
            xscale=False,
            dropout_rate_emb=0.1,
        )
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self.pos_enc.extend_pe(max_pos, device, dtype)


    @staticmethod
    def get_image_processor(model_id):
        return AutoImageProcessor.from_pretrained(f'facebook/{model_id}')

    def forward(self, video_frames: torch.Tensor, video_lengths, attention_mask: torch.Tensor) -> torch.Tensor:
        # video_frames shape: (B, C, T, H, W) -> (B*T, C, H, W)
        B, C, T, H, W = video_frames.shape
        video_frames = video_frames.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        with self.inference_ctx_fn():
            dino_feats = self.dino_model(pixel_values=video_frames).last_hidden_state  # (B*T, #patches, feat_dim)

        dino_feats = dino_feats[:, -self.num_patches:, :]  # Get rid of CLS token.
        dino_feats = dino_feats.reshape(B*T, self.num_patches, dino_feats.shape[-1])  # (B, T, #patches, feat_dim)

        # BxT, 1, D
        pooled_feats = self.attn_pooling(self.attn_pooling_query.repeat((dino_feats.shape[0], 1, 1)), dino_feats, dino_feats, need_weights=False)[0]
        
        # BxTxD
        pooled_feats = pooled_feats.reshape(B, T, dino_feats.shape[-1])

        pooled_feats = self.conv_preproccessing(pooled_feats.permute(0, 2, 1)).permute(0, 2, 1)

        max_vid_length = pooled_feats.shape[1]
        pad_mask, att_mask = self._create_masks(
            att_context_size=[-1, -1],
            padding_length=video_lengths,
            max_audio_length=max_vid_length,
            offset=None,
            device=pooled_feats.device,
        )

        pooled_feats, pos_emb = self.pos_enc(x=pooled_feats, cache_len=0)

        # We need to add pos embeddings to introduce the temporal order and then apply Conformer on top.
        for i, layer in enumerate(self.conformer_layers):    
            pooled_feats = layer(
                x=pooled_feats,
                att_mask=att_mask,
                pos_emb=pos_emb,
                pad_mask=pad_mask,
                cache_last_channel=None,
                cache_last_time=None,
            )

        return BaseModelOutput(
            last_hidden_state=pooled_feats,
            hidden_states=None,
            attentions=None,
        )
        

    # Taken from ConformerEncoder
    def _create_masks(self, att_context_size, padding_length, max_audio_length, offset, device):
        if self.self_attention_model != "rel_pos_local_attn":
            att_mask = torch.ones(1, max_audio_length, max_audio_length, dtype=torch.bool, device=device)

            if self.att_context_style == "regular":
                if att_context_size[0] >= 0:
                    att_mask = att_mask.triu(diagonal=-att_context_size[0])
                if att_context_size[1] >= 0:
                    att_mask = att_mask.tril(diagonal=att_context_size[1])
            elif self.att_context_style == "chunked_limited":
                # When right context is unlimited, just the left side of the masking need to get updated
                if att_context_size[1] == -1:
                    if att_context_size[0] >= 0:
                        att_mask = att_mask.triu(diagonal=-att_context_size[0])
                else:
                    chunk_size = att_context_size[1] + 1
                    # left_chunks_num specifies the number of chunks to be visible by each chunk on the left side
                    if att_context_size[0] >= 0:
                        left_chunks_num = att_context_size[0] // chunk_size
                    else:
                        left_chunks_num = 10000

                    chunk_idx = torch.arange(0, max_audio_length, dtype=torch.int, device=att_mask.device)
                    chunk_idx = torch.div(chunk_idx, chunk_size, rounding_mode="trunc")
                    diff_chunks = chunk_idx.unsqueeze(1) - chunk_idx.unsqueeze(0)
                    chunked_limited_mask = torch.logical_and(
                        torch.le(diff_chunks, left_chunks_num), torch.ge(diff_chunks, 0)
                    )
                    att_mask = torch.logical_and(att_mask, chunked_limited_mask.unsqueeze(0))
        else:
            att_mask = None

        # pad_mask is the masking to be used to ignore paddings
        pad_mask = torch.arange(0, max_audio_length, device=device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(-1)

        if offset is not None:
            pad_mask_off = torch.arange(0, max_audio_length, device=device).expand(
                padding_length.size(0), -1
            ) >= offset.unsqueeze(-1)
            pad_mask = pad_mask_off.logical_and(pad_mask)

        if att_mask is not None:
            # pad_mask_for_att_mask is the mask which helps to ignore paddings
            pad_mask_for_att_mask = pad_mask.unsqueeze(1).repeat([1, max_audio_length, 1])
            pad_mask_for_att_mask = torch.logical_and(pad_mask_for_att_mask, pad_mask_for_att_mask.transpose(1, 2))
            # att_mask is the masking to be used by the MHA layers to ignore the tokens not supposed to be visible
            att_mask = att_mask[:, :max_audio_length, :max_audio_length]
            # paddings should also get ignored, so pad_mask_for_att_mask is used to ignore their corresponding scores
            att_mask = torch.logical_and(pad_mask_for_att_mask, att_mask.to(pad_mask_for_att_mask.device))
            att_mask = ~att_mask

        pad_mask = ~pad_mask
        return pad_mask, att_mask
