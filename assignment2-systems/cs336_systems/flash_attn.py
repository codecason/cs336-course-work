
import torch

class MyTritonFlashAttentionAutogradFunctionClass(torch.autograd.Function):
    def forward(ctx, q, k, v, is_causal):
        """
        Forward pass for FlashAttention2 using Triton kernels.

        Args:
            ctx: Context object to save information for backward pass.
            q: Query tensor of shape (B, n_queries, D).
            k: Key tensor of shape (B, n_queries, D).
            v: Value tensor of shape (B, n_queries, D).
            is_causal: Boolean indicating whether to apply causal masking.
        """
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        return super().backward(ctx, *grad_outputs)


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
                    # NOTE:
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

"""
    version2: q in outer loop
    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, output, logsumexp = ctx.saved_tensors
        grad_q = torch.empty_like(q)
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)

        for q_start in range(0, q.shape[-2], ctx.block_q):
            q_end = min(q_start + ctx.block_q, q.shape[-2])
            q_block = q[..., q_start:q_end, :]
            output_block = output[..., q_start:q_end, :]
            logsumexp_block = logsumexp[..., q_start:q_end]
            grad_output_block = grad_output[..., q_start:q_end, :]
            correction = torch.sum(
                grad_output_block * output_block,
                dim=-1,
                keepdim=True,
            )
            grad_q_block = torch.zeros_like(q_block)

            for k_start in range(0, k.shape[-2], ctx.block_k):
                k_end = min(k_start + ctx.block_k, k.shape[-2])
                k_block = k[..., k_start:k_end, :]
                v_block = v[..., k_start:k_end, :]
                scores = q_block @ k_block.transpose(-2, -1) * ctx.scale

                if ctx.is_causal:
                    q_idx = torch.arange(q_start, q_end, device=q.device)[:, None]
                    k_idx = torch.arange(k_start, k_end, device=q.device)[None, :]
                    causal_mask = q_idx >= k_idx
                    scores = scores.masked_fill(~causal_mask, -1e6)

                probabilities = torch.exp(scores - logsumexp_block.unsqueeze(-1))
                grad_v_block = probabilities.transpose(-2, -1) @ grad_output_block
                grad_v[..., k_start:k_end, :] += grad_v_block

                grad_probabilities = grad_output_block @ v_block.transpose(-2, -1)
                grad_scores = probabilities * (grad_probabilities - correction)
                grad_q_block += grad_scores @ k_block * ctx.scale
                grad_k_block = grad_scores.transpose(-2, -1) @ q_block * ctx.scale
                grad_k[..., k_start:k_end, :] += grad_k_block

            grad_q[..., q_start:q_end, :] = grad_q_block

        return grad_q, grad_k, grad_v, None
"""
