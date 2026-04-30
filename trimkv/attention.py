"""Retention-weighted attention.

TrimKV adds a retention term to the normal attention logits:

    softmax(q.k / sqrt(d) + (t - i) * log(beta_i))

The extra term is just an additive bias. During training it is a full
query-by-key matrix, because every query position has a different token age.
During decoding it is usually a single row for the newest query.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def retention_weighted_attention(
    query: torch.Tensor,       # (B, H_q, T_q, D)
    key: torch.Tensor,         # (B, H_kv, T_k, D)
    value: torch.Tensor,       # (B, H_kv, T_k, D)
    log_scores: torch.Tensor,  # (B, H_kv, T_k) or (B, H_kv, T_q, T_k)
    attn_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Return attention output with shape (B, H_q, T_q, D).

    Grouped-query attention is supported: H_q must be a multiple of H_kv.
    Three-dimensional `log_scores` are broadcast across query positions, which
    is the common one-token decoding case.
    """
    _, h_q, _, head_dim = query.shape
    h_kv = key.shape[1]
    if h_q % h_kv != 0:
        raise ValueError(f"H_q={h_q} must be a multiple of H_kv={h_kv}")

    group_size = h_q // h_kv
    if group_size > 1:
        key = key.repeat_interleave(group_size, dim=1)
        value = value.repeat_interleave(group_size, dim=1)
        log_scores = log_scores.repeat_interleave(group_size, dim=1)

    scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
    logits = torch.matmul(query, key.transpose(-1, -2)) * scale

    if log_scores.dim() == 3:
        logits = logits + log_scores.unsqueeze(2)
    elif log_scores.dim() == 4:
        logits = logits + log_scores
    else:
        raise ValueError("log_scores must be (B, H, T_k) or (B, H, T_q, T_k)")

    if attn_mask is not None:
        logits = logits + attn_mask

    probs = F.softmax(logits, dim=-1, dtype=torch.float32).to(value.dtype)
    return torch.matmul(probs, value)
