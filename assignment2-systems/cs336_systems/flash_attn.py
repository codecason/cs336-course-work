
import torch
import triton
import triton.language as tl
import torch
from einops import rearrange
from triton import cdiv

import math

import torch
import triton
import triton.language as tl



@triton.jit
def _flash_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
    q_stride_1, q_stride_2, q_stride_3,
    k_stride_1, k_stride_2, k_stride_3,
    v_stride_1, v_stride_2, v_stride_3,
    output_stride_1, output_stride_2, output_stride_3,
    logsumexp_stride_1, logsumexp_stride_2,
    softmax_scale,
    L: tl.constexpr,
    D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    q_offsets = row_tile_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    k_offsets_base = tl.arange(0, BLOCK_K)

    d_offsets = tl.arange(0, BLOCK_D) # NOTE

    q_block = tl.load(
        q_ptr +
        batch_idx * q_stride_1 +
        q_offsets[:, None] * q_stride_2 +
        d_offsets[None, :] * q_stride_3,
        mask=((q_offsets[:, None] < L) & (d_offsets[None, :] < D)),
        other=0.0
    )
    # max_values
    m_i = tl.full([BLOCK_Q], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_Q], tl.float32)
    o_i = tl.zeros([BLOCK_Q, BLOCK_D], tl.float32)

    # NOTE: k <= L - 1
    for k_start in tl.range(0, L, BLOCK_K):
        k_offsets = k_start + k_offsets_base

        k_block = tl.load(
            k_ptr +
            batch_idx * k_stride_1 +
            k_offsets[:, None] * k_stride_2 +
            d_offsets[None, :] * k_stride_3,
            mask=((k_offsets[:, None] < L) & (d_offsets[None, :] < D)),
            other=0.0
        )
        v_block = tl.load(
            v_ptr +
            batch_idx * v_stride_1 +
            k_offsets[:, None] * v_stride_2 +
            d_offsets[None, :] * v_stride_3,
            mask=((k_offsets[:, None] < L) & (d_offsets[None, :] < D)),
            other=0.0
        )
        scores = tl.dot(q_block, tl.trans(k_block)) * softmax_scale
        # NOTE
        scores = tl.where(
            k_offsets[None, :] < L,
            scores,
            -float("inf")
        )
        if IS_CAUSAL:
            q_idx = tl.arange(0, BLOCK_Q)[:, None] + row_tile_idx * BLOCK_Q
            k_idx = tl.arange(0, BLOCK_K)[None, :] + k_start
            # k_idx = k_offsets[None, :] < L
            scores = tl.where(q_idx >= k_idx, scores, -float("inf"))

        # m_i, l_i, acc
        m_new = tl.maximum(m_i, tl.max(scores, axis=-1))
        alpha = tl.exp(m_i - m_new)
        probabilities = tl.exp(scores - m_new[:, None])

        l_i = alpha * l_i + tl.sum(probabilities, axis=-1) # NOTE: l_i in (BLOCK_Q,)
        o_i = alpha[:, None] * o_i + tl.dot(probabilities, v_block)

        m_i = m_new

    out = o_i / l_i[:, None]
    lse = m_i + tl.log(l_i) # RECOVER

    tl.store(o_ptr +
             batch_idx * output_stride_1 +
             q_offsets[:, None] * output_stride_2 +
             d_offsets[None, :] * output_stride_3,
             out,
             mask=((q_offsets[:, None] < L) & (d_offsets[None, :] < D)))

    tl.store(lse_ptr +
             batch_idx * logsumexp_stride_1 +
             q_offsets * logsumexp_stride_2,
             lse,
             mask=(q_offsets < L))



@triton.jit
def _flash_backward_kernel(
    q_ptr, k_ptr, v_ptr, delta_ptr, grad_o_ptr, lse_ptr,
    grad_q_ptr, grad_k_ptr, grad_v_ptr,
    q_stride_1, q_stride_2, q_stride_3,
    k_stride_1, k_stride_2, k_stride_3,
    v_stride_1, v_stride_2, v_stride_3,
    delta_stride_1, delta_stride_2,
    grad_output_stride_1, grad_output_stride_2, grad_output_stride_3,
    logsumexp_stride_1, logsumexp_stride_2,
    grad_q_stride_1, grad_q_stride_2, grad_q_stride_3,
    grad_k_stride_1, grad_k_stride_2, grad_k_stride_3,
    grad_v_stride_1, grad_v_stride_2, grad_v_stride_3,
    softmax_scale,
    L: tl.constexpr,
    D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    col_tile_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    q_offsets_base = tl.arange(0, BLOCK_Q)
    k_offsets = col_tile_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    d_offsets = tl.arange(0, BLOCK_D)

    k_block = tl.load(
        k_ptr +
        batch_idx * k_stride_1 +
        k_offsets[:, None] * k_stride_2 +
        d_offsets[None, :] * k_stride_3,
        mask=((k_offsets[:, None] < L) & (d_offsets[None, :] < D)),
        other=0.0
    )
    v_block = tl.load(
        v_ptr +
        batch_idx * v_stride_1 +
        k_offsets[:, None] * v_stride_2 +
        d_offsets[None, :] * v_stride_3,
        mask=((k_offsets[:, None] < L) & (d_offsets[None, :] < D)),
        other=0.0
    )

    grad_k_block = tl.zeros([BLOCK_K, BLOCK_D], tl.float32)
    grad_v_block = tl.zeros([BLOCK_K, BLOCK_D], tl.float32)

    for q_start in tl.range(0, L, BLOCK_Q):
        q_offsets = q_offsets_base + q_start
        q_block = tl.load(
            q_ptr +
            batch_idx * q_stride_1 +
            q_offsets[:, None] * q_stride_2 +
            d_offsets[None, :] * q_stride_3,
            mask=((q_offsets[:, None] < L) & (d_offsets[None, :] < D)),
            other=0.0
        )
        grad_o_block = tl.load(
            grad_o_ptr +
            batch_idx * grad_output_stride_1 +
            q_offsets[:, None] * grad_output_stride_2 +
            d_offsets[None, :] * grad_output_stride_3,
            mask=((q_offsets[:, None] < L) & (d_offsets[None, :] < D)),
            other=0.0
        )

        scores = tl.dot(q_block, tl.trans(k_block)) * softmax_scale
        scores = tl.where(
            ((q_offsets[:, None] < L) & (k_offsets[None, :] < L)),
            scores,
            -float("inf")
        )
        if IS_CAUSAL:
            q_idx = q_offsets[:, None]
            k_idx = k_offsets[None, :]
            scores = tl.where(q_idx >= k_idx, scores, -float("inf"))

        lse = tl.load(
            lse_ptr +
            batch_idx * logsumexp_stride_1 +
            q_offsets * logsumexp_stride_2,
            mask=(q_offsets < L),
            other=0.0
        )
        p_i = tl.exp(scores - lse[:, None])

        grad_v_block += tl.dot(tl.trans(p_i), grad_o_block)
        grad_p_i = tl.dot(grad_o_block, tl.trans(v_block))
        delta = tl.load(
            delta_ptr +
            batch_idx * delta_stride_1 +
            q_offsets * delta_stride_2,
            mask=(q_offsets < L),
            other=0.0
        )
        grad_s_i = p_i * (grad_p_i - delta[:, None]) * softmax_scale

        grad_q_block = tl.dot(grad_s_i, k_block)
        grad_k_block += tl.dot(tl.trans(grad_s_i), q_block)

        tl.atomic_add(
            grad_q_ptr +
            batch_idx * grad_q_stride_1 +
            q_offsets[:, None] * grad_q_stride_2 +
            d_offsets[None, :] * grad_q_stride_3,
            grad_q_block,
            mask=((q_offsets[:, None] < L) & (d_offsets[None, :] < D)),
        )

    tl.store(
        grad_k_ptr +
        batch_idx * grad_k_stride_1 +
        k_offsets[:, None] * grad_k_stride_2 +
        d_offsets[None, :] * grad_k_stride_3,
        grad_k_block,
        mask=((k_offsets[:, None] < L) & (d_offsets[None, :] < D)),
    )
    tl.store(
        grad_v_ptr +
        batch_idx * grad_v_stride_1 +
        k_offsets[:, None] * grad_v_stride_2 +
        d_offsets[None, :] * grad_v_stride_3,
        grad_v_block,
        mask=((k_offsets[:, None] < L) & (d_offsets[None, :] < D)),
    )



class MyTritonFlashAttentionAutogradFunctionClass(torch.autograd.Function):
    BLOCK_Q = 64
    BLOCK_K = 64
    BLOCK_D = 64

    @staticmethod
    def forward(ctx, q, k, v, is_causal):
        """
        Forward pass for FlashAttention using Triton kernel.

        Args:
            ctx: Context object to save information for backward pass.
            q: Query tensor of shape (B, n_queries, D).
            k: Key tensor of shape (B, n_queries, D).
            v: Value tensor of shape (B, n_queries, D).
            is_causal: Boolean indicating whether to apply causal masking.
        """
        B = q.shape[0]
        L = q.shape[1]
        D = q.shape[2]

        BLOCK_Q = MyTritonFlashAttentionAutogradFunctionClass.BLOCK_Q
        BLOCK_K = MyTritonFlashAttentionAutogradFunctionClass.BLOCK_K
        BLOCK_D = MyTritonFlashAttentionAutogradFunctionClass.BLOCK_D

        if D > BLOCK_D:
            raise ValueError(f"D={D} must be <= BLOCK_D={BLOCK_D}")

        output = torch.empty_like(q)
        logsumexp = torch.empty((B, L), device=q.device, dtype=torch.float32)

        grid = (
            triton.cdiv(L, BLOCK_Q),
            B,
        )

        softmax_scale = 1.0 / D ** 0.5

        _flash_fwd_kernel[grid](
            q, k, v,
            output, logsumexp,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            logsumexp.stride(0), logsumexp.stride(1),
            softmax_scale, L, D, IS_CAUSAL=is_causal,
            BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
            num_warps=4 # NOTE: num_warps
        )

        ctx.save_for_backward(q, k, v, output, logsumexp)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, o, lse = ctx.saved_tensors
        is_causal = ctx.is_causal

        B = q.shape[0]
        L = q.shape[1]
        D = q.shape[2]

        BLOCK_Q = 32
        BLOCK_K = 32
        BLOCK_D = MyTritonFlashAttentionAutogradFunctionClass.BLOCK_D

        if D > BLOCK_D:
            raise ValueError(f"D={D} must be <= BLOCK_D={BLOCK_D}")

        grid = (
            triton.cdiv(L, BLOCK_K),
            B,
        )
        softmax_scale = 1.0 / D ** 0.5

        grad_q = torch.zeros_like(q)
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)
        delta = torch.sum(grad_output * o, dim=-1)

        _flash_backward_kernel[grid](
            q, k, v, delta, grad_output, lse,
            grad_q, grad_k, grad_v,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            delta.stride(0), delta.stride(1),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
            lse.stride(0), lse.stride(1),
            grad_q.stride(0), grad_q.stride(1), grad_q.stride(2),
            grad_k.stride(0), grad_k.stride(1), grad_k.stride(2),
            grad_v.stride(0), grad_v.stride(1), grad_v.stride(2),
            softmax_scale, L, D, IS_CAUSAL=is_causal,
            BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
            num_warps=4 # NOTE: num_warps
        )

        return grad_q, grad_k, grad_v, None


class NaiveFlashAttentionAutogradFunctionClass(torch.autograd.Function):
    BLOCK_Q = 64
    BLOCK_K = 64

    @staticmethod
    def forward(ctx, q, k, v, is_causal):
        """
        Forward pass for Naive FlashAttention using PyTorch operations.

        Args:
            ctx: Context object to save information for backward pass.
            q: Query tensor of shape (B, L, D).
            k: Key tensor of shape (B, L, D).
            v: Value tensor of shape (B, L, D).
            is_causal: Boolean indicating whether to apply causal masking.
        """
        n_queries = q.shape[1]
        d = q.shape[-1]
        b_q = NaiveFlashAttentionAutogradFunctionClass.BLOCK_Q
        b_k = NaiveFlashAttentionAutogradFunctionClass.BLOCK_K
        k_blocks = n_queries // b_k
        output = torch.empty(
            (*q.shape[:-2], n_queries, d),
            device=q.device,
            dtype=q.dtype
        )
        logsumexp = torch.empty(q.shape[:-1], device=q.device, dtype=q.dtype)

        for q_start in range(0, n_queries, b_q):
            q_end = min(q_start + b_q, n_queries)
            q_block = q[:, q_start:q_end, :]
            block_shape = q_block.shape[:-1]
            # NOTE: (q_k, d) -> (*block_shape, d)
            m_i = torch.full(block_shape, -1e9, device=q.device, dtype=q.dtype)
            o_i = torch.zeros((*block_shape, d), device=q.device, dtype=q.dtype)
            # l_i = torch.zeros_like(m_i)
            l_i = torch.zeros(block_shape, device=q.device, dtype=q.dtype)

            for j in range(k_blocks):
                k_block = k[:, j*b_k:(j+1)*b_k, :]
                v_block = v[:, j*b_k:(j+1)*b_k, :]

                scores = q_block @ k_block.transpose(-2, -1) / (q.shape[-1] ** 0.5)
                if is_causal:
                    q_idx = torch.arange(q_start, q_end, device=q.device, dtype=q.dtype)[:, None]
                    k_idx = torch.arange(j*b_k, j*b_k + b_k, device=q.device, dtype=q.dtype)[None, :]
                    scores = torch.masked_fill(scores, ~(q_idx >= k_idx), -1e9)

                block_m = torch.max(scores, dim=-1).values
                m_new = torch.maximum(m_i, block_m)

                exp = torch.exp(m_i - m_new)
                p_i = torch.exp(scores - m_new.unsqueeze(-1))
                l_i = exp * l_i + torch.sum(p_i, dim=-1)

                o_i = exp.unsqueeze(-1) * o_i + p_i @ v_block
                m_i = m_new

            output[..., q_start:q_end, :] = o_i / l_i.unsqueeze(-1)
            logsumexp[..., q_start:q_end] = m_i + torch.log(l_i)

        ctx.save_for_backward(q, k, v, output, logsumexp)
        ctx.is_causal = is_causal
        ctx.b_k = b_k
        ctx.b_q = b_q
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, output, logsumexp = ctx.saved_tensors
        # grad_output.shape == (..., n_queries, d_v)

        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)
        b_q = ctx.b_q
        b_k = ctx.b_k

        assist_d = torch.sum(grad_output * output, dim=-1)
        scale = q.shape[-1] ** -0.5
        n_queries = q.shape[-2]
        n_keys = k.shape[-2]

        for k_start in range(0, n_keys, b_k):
            k_end = min(k_start + b_k, n_keys)
            k_block = k[..., k_start:k_end, :]
            v_block = v[..., k_start:k_end, :]

            for q_start in range(0, n_queries, b_q):
                q_end = min(q_start + b_q, n_queries)
                q_block = q[..., q_start:q_end, :]
                grad_o_block = grad_output[..., q_start:q_end, :]
                scores = q_block @ k_block.transpose(-1, -2) * scale

                if ctx.is_causal:
                    q_idx = torch.arange(q_start, q_end, device=q.device)[:, None]
                    k_idx = torch.arange(k_start, k_end, device=q.device)[None, :]
                    scores = scores.masked_fill(~(q_idx >= k_idx), -1e9)

                l_i = logsumexp[..., q_start:q_end]
                p_i = torch.exp(scores - l_i.unsqueeze(-1))

                grad_v[..., k_start:k_end, :] += p_i.transpose(-1, -2) @ grad_o_block
                grad_p_i = grad_o_block @ v_block.transpose(-1, -2)
                grad_s_i = p_i * (grad_p_i - assist_d[..., q_start:q_end].unsqueeze(-1)) * scale
                # NOTE: must use atomic
                grad_q[..., q_start:q_end, :] += grad_s_i @ k_block
                grad_k[..., k_start:k_end, :] += grad_s_i.transpose(-1, -2) @ q_block

        return grad_q, grad_k, grad_v, None
