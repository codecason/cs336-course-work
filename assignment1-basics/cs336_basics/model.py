from __future__ import annotations

import math

import torch
from torch import Tensor


def linear(weights: Tensor, in_features: Tensor) -> Tensor:
    return in_features @ weights.T


def embedding(weights: Tensor, token_ids: Tensor) -> Tensor:
    return weights[token_ids]


def silu(in_features: Tensor) -> Tensor:
    return torch.sigmoid(in_features) * in_features


def swiglu(w1_weight: Tensor, w2_weight: Tensor, w3_weight: Tensor, in_features: Tensor) -> Tensor:
    gate = silu(linear(w1_weight, in_features))
    value = linear(w3_weight, in_features)
    return linear(w2_weight, gate * value)


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def rmsnorm(weights: Tensor, in_features: Tensor, eps: float) -> Tensor:
    mean_square = in_features.pow(2).mean(dim=-1, keepdim=True)
    return in_features * torch.rsqrt(mean_square + eps) * weights


def rope(
    in_query_or_key: Tensor,
    token_positions: Tensor,
    theta: float,
    max_seq_len: int,
) -> Tensor:
    d_k = in_query_or_key.shape[-1]
    if d_k % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")
    if token_positions.numel() and int(token_positions.max()) >= max_seq_len:
        raise ValueError("token position exceeds max_seq_len")
    device = in_query_or_key.device
    half_dim = d_k // 2
    positions = -torch.arange(0, half_dim, device=device, dtype=torch.float32)
    frequencies = theta ** (positions / half_dim)
    angles = token_positions.to(device=device, dtype=torch.float32).unsqueeze(-1) * frequencies
    cos = angles.cos()
    sin = angles.sin()

    even = in_query_or_key[..., 0::2]
    odd = in_query_or_key[..., 1::2]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


def _causal_mask(sequence_length: int, device: torch.device) -> Tensor:
    return torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device).tril()


def multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
    theta: float | None = None,
    max_seq_len: int | None = None,
    token_positions: Tensor | None = None,
) -> Tensor:
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    sequence_length = in_features.shape[-2]
    head_dim = d_model // num_heads

    q = linear(q_proj_weight, in_features)
    k = linear(k_proj_weight, in_features)
    v = linear(v_proj_weight, in_features)

    q = q.reshape(*q.shape[:-1], num_heads, head_dim).movedim(-2, -3)
    k = k.reshape(*k.shape[:-1], num_heads, head_dim).movedim(-2, -3)
    v = v.reshape(*v.shape[:-1], num_heads, head_dim).movedim(-2, -3)

    # NOTE: rope may not be used
    if theta is not None:
        if max_seq_len is None:
            raise ValueError("max_seq_len is required when theta is provided")
        if token_positions is None:
            token_positions = torch.arange(sequence_length, device=in_features.device)
        q = rope(q, token_positions, theta, max_seq_len)
        k = rope(k, token_positions, theta, max_seq_len)

    attention = scaled_dot_product_attention(q, k, v, _causal_mask(sequence_length, in_features.device))
    attention = attention.movedim(-3, -2).reshape(*in_features.shape[:-1], d_model)
    return linear(o_proj_weight, attention)


def transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Tensor,
) -> Tensor:
    normalized = rmsnorm(weights["ln1.weight"], in_features, 1e-5)
    attention = multihead_self_attention(
        d_model,
        num_heads,
        weights["attn.q_proj.weight"],
        weights["attn.k_proj.weight"],
        weights["attn.v_proj.weight"],
        weights["attn.output_proj.weight"],
        normalized,
        theta=theta,
        max_seq_len=max_seq_len,
    )
    residual = in_features + attention

    normalized = rmsnorm(weights["ln2.weight"], residual, 1e-5)
    feedforward = swiglu(
        weights["ffn.w1.weight"],
        weights["ffn.w2.weight"],
        weights["ffn.w3.weight"],
        normalized,
    )
    return residual + feedforward


def transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Tensor,
) -> Tensor:
    features = embedding(weights["token_embeddings.weight"], in_indices)
    for layer in range(num_layers):
        layer_weights = {
            key.removeprefix(f"layers.{layer}."): value
            for key, value in weights.items()
            if key.startswith(f"layers.{layer}.")
        }
        features = transformer_block(
            d_model,
            num_heads,
            d_ff,
            context_length,
            rope_theta,
            layer_weights,
            features,
        )
    features = rmsnorm(weights["ln_final.weight"], features, 1e-5)
    return linear(weights["lm_head.weight"], features)
