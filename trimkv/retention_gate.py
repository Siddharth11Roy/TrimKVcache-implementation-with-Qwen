"""Retention gate used by TrimKV.

The gate maps each layer input hidden state to one retention score per KV head.
Scores are in (0, 1), so they can be used as the base of the exponential decay
term described in the paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RetentionGate(nn.Module):
    """Small per-layer MLP that predicts token retention scores."""

    def __init__(
        self,
        hidden_size: int,
        num_kv_heads: int,
        gate_hidden: int = 512,
        init_bias: float = 18.0,
    ) -> None:
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, num_kv_heads),
        )

        # Start near full retention. Training can then learn where forgetting is
        # useful instead of damaging the base model immediately.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, init_bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map (B, T, D) hidden states to (B, T, H_kv) scores."""
        logits = self.mlp(hidden)
        return torch.sigmoid(logits.clamp(min=-30.0, max=30.0))
