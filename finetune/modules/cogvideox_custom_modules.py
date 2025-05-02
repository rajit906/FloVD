import math
from typing import List, Optional, Tuple, Union, Dict, Any
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from diffusers import CogVideoXTransformer3DModel
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock
from diffusers.models.normalization import CogVideoXLayerNormZero
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0, Attention
from diffusers.models.embeddings import CogVideoXPatchEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph

from contextlib import contextmanager
from peft.tuners.lora.layer import LoraLayer  # PEFT의 LoRA 레이어 기본 클래스

import pdb

# Code heavily borrowed from https://github.com/huggingface/diffusers


class enable_lora:
    def __init__(self, modules, enable=True):
        self.modules = modules
        self.enable = enable
        self.prev_states = {}
    
    def __enter__(self):
        for module in self.modules:
            self.prev_states[module] = getattr(module, "lora_enabled", True)
            setattr(module, "lora_enabled", self.enable)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for module in self.modules:
            setattr(module, "lora_enabled", self.prev_states[module])
        return False



class CustomCogVideoXPatchEmbed(CogVideoXPatchEmbed):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        patch_size = kwargs['patch_size']
        patch_size_t = kwargs['patch_size_t']
        bias = kwargs['bias']
        in_channels = kwargs['in_channels']
        embed_dim = kwargs['embed_dim']
        
        # projection layer for flow latents
        if patch_size_t is None:
            # CogVideoX 1.0 checkpoints
            self.flow_proj = nn.Conv2d(in_channels//2, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias)
        else:
            # CogVideoX 1.5 checkpoints
            self.flow_proj = nn.Linear(in_channels//2 * patch_size * patch_size * patch_size_t, embed_dim)
        
        # Add positional embedding for flow_embeds
        if self.use_positional_embeddings or self.use_learned_positional_embeddings:
            flow_pos_embedding = self._get_positional_embeddings(self.sample_height, self.sample_width, self.sample_frames)[:,self.max_text_seq_length:] # shape: [1, 17550, 3072]
            self.flow_pos_embedding = nn.Parameter(flow_pos_embedding)
        
    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor, flow_embeds: torch.Tensor):
        r"""
        Args:
            text_embeds (`torch.Tensor`):
                Input text embeddings. Expected shape: (batch_size, seq_length, embedding_dim).
            image_embeds (`torch.Tensor`):
                Input image embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
            flow_embeds (`torch.Tensor`):
                Input flow embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
        """
        text_embeds = self.text_proj(text_embeds)

        batch_size, num_frames, channels, height, width = image_embeds.shape

        if self.patch_size_t is None:
            # embed video latents
            image_embeds = image_embeds.reshape(-1, channels, height, width)
            image_embeds = self.proj(image_embeds)
            image_embeds = image_embeds.view(batch_size, num_frames, *image_embeds.shape[1:])
            image_embeds = image_embeds.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
            image_embeds = image_embeds.flatten(1, 2)  # [batch, num_frames x height x width, channels]
            
            # embed flow latents
            flow_embeds = flow_embeds.reshape(-1, channels//2, height, width)
            flow_embeds = self.flow_proj(flow_embeds)
            flow_embeds = flow_embeds.view(batch_size, num_frames, *flow_embeds.shape[1:])
            flow_embeds = flow_embeds.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
            flow_embeds = flow_embeds.flatten(1, 2)  # [batch, num_frames x height x width, channels]    
        else:
            p = self.patch_size
            p_t = self.patch_size_t

            # embed video latents
            image_embeds = image_embeds.permute(0, 1, 3, 4, 2)
            image_embeds = image_embeds.reshape(
                batch_size, num_frames // p_t, p_t, height // p, p, width // p, p, channels
            )
            image_embeds = image_embeds.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(4, 7).flatten(1, 3)
            image_embeds = self.proj(image_embeds)
            
            # embed flow latents
            flow_embeds = flow_embeds.permute(0, 1, 3, 4, 2)
            flow_embeds = flow_embeds.reshape(
                batch_size, num_frames // p_t, p_t, height // p, p, width // p, p, channels//2
            )
            flow_embeds = flow_embeds.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(4, 7).flatten(1, 3)
            flow_embeds = self.flow_proj(flow_embeds)
        
        # Curriculum learning of flow token
        # flow_embeds = self.flow_scale * flow_embeds


        embeds = torch.cat(
            [text_embeds, image_embeds, flow_embeds], dim=1
        ).contiguous()  # [batch, num_frames x height x width + seq_length + num_frames x height x width, channels]

        if self.use_positional_embeddings or self.use_learned_positional_embeddings:
            if self.use_learned_positional_embeddings and (self.sample_width != width or self.sample_height != height):
                raise ValueError(
                    "It is currently not possible to generate videos at a different resolution that the defaults. This should only be the case with 'THUDM/CogVideoX-5b-I2V'."
                    "If you think this is incorrect, please open an issue at https://github.com/huggingface/diffusers/issues."
                )

            pre_time_compression_frames = (num_frames - 1) * self.temporal_compression_ratio + 1

            if (
                self.sample_height != height
                or self.sample_width != width
                or self.sample_frames != pre_time_compression_frames
            ):
                pos_embedding = self._get_positional_embeddings(
                    height, width, pre_time_compression_frames, device=embeds.device
                )
            else:
                pos_embedding = self.pos_embedding

            # Previous version..                  
            # pos_embedding = pos_embedding.to(dtype=embeds.dtype)
            # embeds = embeds + pos_embedding
            
            # Add flow embedding..
            # flow_pos_embedding = self.flow_pos_scale * self.flow_pos_embedding
            flow_pos_embedding = self.flow_pos_embedding
            pos_embedding_total = torch.cat([pos_embedding, flow_pos_embedding], dim=1).to(dtype=embeds.dtype)
            embeds = embeds + pos_embedding_total

        return embeds



@maybe_allow_in_graph
class CustomCogVideoXBlock(nn.Module):
    r"""
    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CustomCogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)
        attention_kwargs = attention_kwargs or {}

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **attention_kwargs,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states


class CustomCogVideoXAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        notextinflow: Optional[bool] = False,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        if not notextinflow:
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            if not notextinflow:
                query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
                if not attn.is_cross_attention:
                    key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)
            else:
                query[:, :, :] = apply_rotary_emb(query[:, :, :], image_rotary_emb)
                if not attn.is_cross_attention:
                    key[:, :, :] = apply_rotary_emb(key[:, :, :], image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if not notextinflow:
            encoder_hidden_states, hidden_states = hidden_states.split(
                [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
            )
            
        return hidden_states, encoder_hidden_states