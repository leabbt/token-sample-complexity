"""Chunked attention computing only the per-token output statistics (mean, cov)
for sequences too large to hold the full attention matrix in memory."""

from __future__ import annotations

import math
from typing import Tuple

import torch


def _transpose_for_scores(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    b = x.size(0)
    return x.view(b, -1, num_heads, head_dim).permute(0, 2, 1, 3)


def chunked_attention_memory_efficient(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    chunk_size: int = 512,
    attention_head_size: int = 64,
    all_head_size: int = 768,
    use_fp16: bool = True,
    temperature_scale: float = 1.0,
    acc_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Online-softmax attention that chunks both Q and K.

    Q, K, V have shape [batch, seq_len, hidden_size]. Returns (mean, cov) of the
    per-token attention output, each computed in `acc_dtype`. Memory cost is
    O(chunk_size**2) instead of O(seq_len * chunk_size).
    """
    batch_size, seq_len, hidden_size = Q.shape
    num_heads = all_head_size // attention_head_size
    effective_scale = temperature_scale / math.sqrt(attention_head_size)

    orig_dtype = Q.dtype
    if use_fp16 and Q.dtype != torch.float16:
        Q, K, V = Q.half(), K.half(), V.half()

    sum_y = torch.zeros((batch_size, hidden_size), device=Q.device, dtype=acc_dtype)
    sum_yy = torch.zeros((batch_size, hidden_size, hidden_size), device=Q.device, dtype=acc_dtype)

    num_q = (seq_len + chunk_size - 1) // chunk_size
    num_k = num_q

    for qi in range(num_q):
        qs, qe = qi * chunk_size, min((qi + 1) * chunk_size, seq_len)
        q_len = qe - qs
        Q_chunk = _transpose_for_scores(Q[:, qs:qe, :], num_heads, attention_head_size)

        m = torch.full((batch_size, num_heads, q_len, 1), float("-inf"),
                       device=Q.device, dtype=Q.dtype)
        l = torch.zeros((batch_size, num_heads, q_len, 1), device=Q.device, dtype=Q.dtype)
        o = torch.zeros((batch_size, num_heads, q_len, attention_head_size),
                        device=Q.device, dtype=Q.dtype)

        for ki in range(num_k):
            ks, ke = ki * chunk_size, min((ki + 1) * chunk_size, seq_len)
            K_chunk = _transpose_for_scores(K[:, ks:ke, :], num_heads, attention_head_size)
            V_chunk = _transpose_for_scores(V[:, ks:ke, :], num_heads, attention_head_size)

            scores = torch.matmul(Q_chunk, K_chunk.transpose(-1, -2)) * effective_scale
            m_chunk = scores.max(dim=-1, keepdim=True).values
            m_new = torch.maximum(m, m_chunk)
            exp_diff = torch.exp(m - m_new)
            l = l * exp_diff
            o = o * exp_diff
            exp_scores = torch.exp(scores - m_new)
            l = l + exp_scores.sum(dim=-1, keepdim=True)
            o = o + torch.matmul(exp_scores, V_chunk)
            m = m_new

        y_block = (o / l).permute(0, 2, 1, 3).contiguous().view(batch_size, q_len, hidden_size)
        y_block = y_block.to(acc_dtype)
        sum_y += y_block.sum(dim=1)
        sum_yy += y_block.transpose(1, 2) @ y_block

        if Q.is_cuda:
            torch.cuda.empty_cache()

    mean = sum_y / seq_len
    cov = (sum_yy / seq_len) - mean.unsqueeze(2) @ mean.unsqueeze(1)
    cov = 0.5 * (cov + cov.transpose(-1, -2))

    if use_fp16 and orig_dtype != torch.float16:
        mean = mean.to(orig_dtype)
        cov = cov.to(orig_dtype)
    return mean, cov


def chunked_attention_cpu_offload(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    chunk_size: int = 512,
    attention_head_size: int = 64,
    all_head_size: int = 768,
    temperature_scale: float = 1.0,
    acc_dtype: torch.dtype = torch.float32,
    compute_device: str | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CPU-offload variant: keeps K, V on host RAM, streams chunks to GPU.

    Slower than `chunked_attention_memory_efficient` but works when even the
    chunk accumulators don't fit on the device.
    """
    batch_size, seq_len, hidden_size = Q.shape
    num_heads = all_head_size // attention_head_size
    effective_scale = temperature_scale / math.sqrt(attention_head_size)

    if compute_device is None:
        device = Q.device
    else:
        device = torch.device(compute_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")

    K_heads = _transpose_for_scores(K, num_heads, attention_head_size).cpu()
    V_heads = _transpose_for_scores(V, num_heads, attention_head_size).cpu()
    Q_cpu = Q.cpu()
    del K, V
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sum_y = torch.zeros((batch_size, hidden_size), device="cpu", dtype=acc_dtype)
    sum_yy = torch.zeros((batch_size, hidden_size, hidden_size), device="cpu", dtype=acc_dtype)

    num_q = (seq_len + chunk_size - 1) // chunk_size
    num_k = num_q

    for qi in range(num_q):
        qs, qe = qi * chunk_size, min((qi + 1) * chunk_size, seq_len)
        q_len = qe - qs
        Q_chunk = _transpose_for_scores(Q_cpu[:, qs:qe, :].to(device),
                                        num_heads, attention_head_size)

        m = torch.full((batch_size, num_heads, q_len, 1), float("-inf"),
                       device=device, dtype=Q_chunk.dtype)
        l = torch.zeros((batch_size, num_heads, q_len, 1), device=device, dtype=Q_chunk.dtype)
        o = torch.zeros((batch_size, num_heads, q_len, attention_head_size),
                        device=device, dtype=Q_chunk.dtype)

        for ki in range(num_k):
            ks, ke = ki * chunk_size, min((ki + 1) * chunk_size, seq_len)
            K_chunk = K_heads[:, :, ks:ke, :].to(device)
            V_chunk = V_heads[:, :, ks:ke, :].to(device)
            scores = torch.matmul(Q_chunk, K_chunk.transpose(-1, -2)) * effective_scale
            m_chunk = scores.max(dim=-1, keepdim=True).values
            m_new = torch.maximum(m, m_chunk)
            exp_diff = torch.exp(m - m_new)
            l = l * exp_diff
            o = o * exp_diff
            exp_scores = torch.exp(scores - m_new)
            l = l + exp_scores.sum(dim=-1, keepdim=True)
            o = o + torch.matmul(exp_scores, V_chunk)
            m = m_new
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        y_block = (o / l).permute(0, 2, 1, 3).contiguous().view(batch_size, q_len, hidden_size)
        y_block_cpu = y_block.to(dtype=acc_dtype, device="cpu")
        sum_y += y_block_cpu.sum(dim=1)
        sum_yy += y_block_cpu.transpose(1, 2) @ y_block_cpu

    mean = sum_y / seq_len
    cov = (sum_yy / seq_len) - mean.unsqueeze(2) @ mean.unsqueeze(1)
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    return mean.to(device), cov.to(device)


def chunked_attention_auto(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    chunk_size: int | None = None,
    attention_head_size: int = 64,
    all_head_size: int = 768,
    force_cpu_offload: bool = False,
    temperature_scale: float = 1.0,
    acc_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pick the best chunked-attention method given available memory."""
    if force_cpu_offload:
        return chunked_attention_cpu_offload(
            Q, K, V, chunk_size=512,
            attention_head_size=attention_head_size, all_head_size=all_head_size,
            temperature_scale=temperature_scale, acc_dtype=acc_dtype,
        )

    if torch.cuda.is_available():
        free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        free_gb = free / (1024**3)
    else:
        free_gb = 0

    if chunk_size is None:
        if free_gb > 10:
            chunk_size = 1024
        elif free_gb > 5:
            chunk_size = 512
        elif free_gb > 2:
            chunk_size = 256
        else:
            chunk_size = 128

    try:
        return chunked_attention_memory_efficient(
            Q, K, V, chunk_size=chunk_size,
            attention_head_size=attention_head_size, all_head_size=all_head_size,
            use_fp16=True, temperature_scale=temperature_scale, acc_dtype=acc_dtype,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return chunked_attention_cpu_offload(
            Q, K, V, chunk_size=256,
            attention_head_size=attention_head_size, all_head_size=all_head_size,
            temperature_scale=temperature_scale, acc_dtype=acc_dtype,
        )


def get_mean_cov(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean and (population) covariance of a [batch, n, d] tensor along dim=1."""
    n = x.size(1)
    mean = x.mean(dim=1)
    cov = (x.transpose(1, 2) @ x) / n - mean.unsqueeze(2) @ mean.unsqueeze(1)
    return mean, cov


def cross_attention_chunked(
    Q_full: torch.Tensor,
    K_sub: torch.Tensor,
    V_sub: torch.Tensor,
    *,
    attention_head_size: int = 64,
    all_head_size: int = 768,
    temperature_scale: float = 1.0,
    q_chunk_size: int = 4096,
) -> torch.Tensor:
    """`Y_sub[i] = softmax(Q_full[i] · K_sub^T / √d) · V_sub`, chunked over Q.

    Used by the window experiment to compute attention of every query against
    a subset of keys/values.
    """
    batch_size, N, _ = Q_full.shape
    num_heads = all_head_size // attention_head_size
    scale = temperature_scale / math.sqrt(attention_head_size)

    K_heads = _transpose_for_scores(K_sub, num_heads, attention_head_size)
    V_heads = _transpose_for_scores(V_sub, num_heads, attention_head_size)

    Y = torch.empty(batch_size, N, all_head_size, device=Q_full.device, dtype=Q_full.dtype)
    num_chunks = (N + q_chunk_size - 1) // q_chunk_size
    for qi in range(num_chunks):
        qs, qe = qi * q_chunk_size, min((qi + 1) * q_chunk_size, N)
        Q_chunk = _transpose_for_scores(Q_full[:, qs:qe, :], num_heads, attention_head_size)
        scores = torch.matmul(Q_chunk, K_heads.transpose(-1, -2)) * scale
        probs = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, V_heads).permute(0, 2, 1, 3).contiguous()
        Y[:, qs:qe, :] = ctx.view(batch_size, -1, all_head_size)
    return Y


def full_attention_output_chunked(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    attention_head_size: int = 64,
    all_head_size: int = 768,
    temperature_scale: float = 1.0,
    q_chunk_size: int = 2048,
    k_chunk_size: int = 4096,
) -> torch.Tensor:
    """Exact `Y_full = softmax(Q K^T / √d) V`, online-softmax chunked over both axes."""
    batch_size, N, _ = Q.shape
    num_heads = all_head_size // attention_head_size
    scale = temperature_scale / math.sqrt(attention_head_size)

    Y = torch.empty(batch_size, N, all_head_size, device=Q.device, dtype=torch.float32)
    num_q = (N + q_chunk_size - 1) // q_chunk_size
    num_k = (N + k_chunk_size - 1) // k_chunk_size

    for qi in range(num_q):
        qs, qe = qi * q_chunk_size, min((qi + 1) * q_chunk_size, N)
        q_len = qe - qs
        Q_chunk = _transpose_for_scores(Q[:, qs:qe, :], num_heads, attention_head_size)
        m = torch.full((batch_size, num_heads, q_len, 1), float("-inf"),
                       device=Q.device, dtype=Q.dtype)
        l = torch.zeros((batch_size, num_heads, q_len, 1), device=Q.device, dtype=Q.dtype)
        o = torch.zeros((batch_size, num_heads, q_len, attention_head_size),
                        device=Q.device, dtype=Q.dtype)
        for ki in range(num_k):
            ks, ke = ki * k_chunk_size, min((ki + 1) * k_chunk_size, N)
            K_chunk = _transpose_for_scores(K[:, ks:ke, :], num_heads, attention_head_size)
            V_chunk = _transpose_for_scores(V[:, ks:ke, :], num_heads, attention_head_size)
            scores = torch.matmul(Q_chunk, K_chunk.transpose(-1, -2)) * scale
            m_chunk = scores.max(dim=-1, keepdim=True).values
            m_new = torch.maximum(m, m_chunk)
            exp_diff = torch.exp(m - m_new)
            l = l * exp_diff
            o = o * exp_diff
            exp_scores = torch.exp(scores - m_new)
            l = l + exp_scores.sum(dim=-1, keepdim=True)
            o = o + torch.matmul(exp_scores, V_chunk)
            m = m_new
        out = (o / l).permute(0, 2, 1, 3).contiguous().view(batch_size, q_len, all_head_size)
        Y[:, qs:qe, :] = out.float()
    return Y
