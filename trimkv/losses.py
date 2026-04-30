"""Training losses for TrimKV."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    ce_weight: float = 0.5,
) -> torch.Tensor:
    """KL from teacher distribution to student, plus optional next-token CE."""
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    kl = kl * (temperature ** 2)

    if labels is None or ce_weight == 0.0:
        return kl

    ce = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )
    return kl + ce_weight * ce


def capacity_loss(
    betas_per_layer: list[torch.Tensor],
    memory_size: int,
) -> torch.Tensor:
    """Penalize expected occupancy above the memory budget.

    For every step t, each earlier token i contributes beta_i ** (t - i). The
    summed contribution is a soft estimate of how full the KV cache would be.
    """
    total = 0.0
    for beta in betas_per_layer:
        _, seq_len, _ = beta.shape
        idx = torch.arange(seq_len, device=beta.device)
        age = (idx.view(-1, 1) - idx.view(1, -1)).clamp(min=0).to(beta.dtype)
        causal = (idx.view(-1, 1) >= idx.view(1, -1)).to(beta.dtype)

        log_beta = beta.clamp(min=1e-8).log()
        log_scores = log_beta.unsqueeze(1) * age.unsqueeze(0).unsqueeze(-1)
        scores = log_scores.exp() * causal.unsqueeze(0).unsqueeze(-1)

        occupancy = scores.sum(dim=2)
        overflow = (occupancy - memory_size).clamp(min=0.0)
        steps = (torch.arange(seq_len, device=beta.device) + 1).to(beta.dtype)
        total = total + (overflow / steps.view(1, -1, 1)).mean()

    return total / max(1, len(betas_per_layer))
