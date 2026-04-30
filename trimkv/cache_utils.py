"""Memory-bounded KV cache used by TrimKV.

Each cached token stores the log of its retention score, `log(beta_i)`, and the
step at which it was created. At step `t`, its current score is:

    beta_i ** (t - i)

When a layer grows past the memory budget, the cache keeps the highest-scoring
tokens and preserves their original temporal order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class _LayerState:
    keys: Optional[torch.Tensor] = None          # (B, H_kv, T_cached, D)
    values: Optional[torch.Tensor] = None        # (B, H_kv, T_cached, D)
    log_beta: Optional[torch.Tensor] = None      # (B, H_kv, T_cached)
    creation_step: Optional[torch.Tensor] = None # (B, T_cached)


class TrimKVCache:
    """Retention-score cache with a fixed per-layer token budget."""

    def __init__(self, num_layers: int, memory_size: int, buffer_size: int = 0) -> None:
        if memory_size <= 0:
            raise ValueError("memory_size must be positive")
        if buffer_size < 0:
            raise ValueError("buffer_size must be non-negative")

        self.num_layers = num_layers
        self.memory_size = memory_size
        self.buffer_size = buffer_size
        self._layers: List[_LayerState] = [_LayerState() for _ in range(num_layers)]
        self._layer_steps: List[int] = [0 for _ in range(num_layers)]
        self._seen_tokens: int = 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        state = self._layers[layer_idx]
        return 0 if state.keys is None else state.keys.shape[-2]

    def current_step(self, layer_idx: int = 0) -> int:
        return self._layer_steps[layer_idx]

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """Compatibility hook used by recent HuggingFace attention masks."""
        return self.get_seq_length(layer_idx) + cache_position.shape[0], 0

    def reset(self) -> None:
        self._layers = [_LayerState() for _ in range(self.num_layers)]
        self._layer_steps = [0 for _ in range(self.num_layers)]
        self._seen_tokens = 0

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        beta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Append new keys/values for one layer and return the retained cache."""
        if key.dim() != 4 or value.dim() != 4:
            raise ValueError("key/value must be shaped (B, H_kv, T, D)")

        batch_size, _, new_tokens, _ = key.shape
        log_beta_new = beta.transpose(1, 2).clamp(min=1e-8).log()

        step0 = self._layer_steps[layer_idx]
        creation_new = torch.arange(
            step0,
            step0 + new_tokens,
            device=key.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(batch_size, -1)

        self._layer_steps[layer_idx] += new_tokens
        self._seen_tokens = max(self._seen_tokens, self._layer_steps[layer_idx])

        state = self._layers[layer_idx]
        if state.keys is None:
            state.keys = key
            state.values = value
            state.log_beta = log_beta_new
            state.creation_step = creation_new
        else:
            state.keys = torch.cat([state.keys, key], dim=-2)
            state.values = torch.cat([state.values, value], dim=-2)
            state.log_beta = torch.cat([state.log_beta, log_beta_new], dim=-1)
            state.creation_step = torch.cat([state.creation_step, creation_new], dim=-1)

        self._enforce_budget(layer_idx)
        return state.keys, state.values, state.log_beta

    def current_log_scores(self, layer_idx: int, step: Optional[int] = None) -> torch.Tensor:
        """Return log retention scores for the retained tokens in one layer."""
        state = self._layers[layer_idx]
        if state.keys is None:
            raise ValueError("cannot score an empty cache")

        step = self._layer_steps[layer_idx] if step is None else step
        age = (step - state.creation_step).clamp(min=0).to(state.log_beta.dtype)
        return state.log_beta * age.unsqueeze(1)

    def _enforce_budget(self, layer_idx: int) -> None:
        state = self._layers[layer_idx]
        if state.keys is None:
            return

        cached_tokens = state.keys.shape[-2]
        if cached_tokens <= self.memory_size + self.buffer_size:
            return

        log_scores = self.current_log_scores(layer_idx)
        token_scores = log_scores.amin(dim=1)
        keep_idx = token_scores.topk(
            self.memory_size,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices
        keep_idx, _ = keep_idx.sort(dim=-1)

        state.keys = _gather_cached(state.keys, keep_idx)
        state.values = _gather_cached(state.values, keep_idx)
        state.log_beta = _gather_cached(state.log_beta, keep_idx)
        state.creation_step = _gather_cached(state.creation_step, keep_idx)


def _gather_cached(tensor: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather batch-specific token indices from cached tensors."""
    if tensor.dim() == 4:
        batch_size, heads, _, head_dim = tensor.shape
        expanded = idx.view(batch_size, 1, -1, 1).expand(batch_size, heads, -1, head_dim)
        return tensor.gather(dim=2, index=expanded)

    if tensor.dim() == 3:
        batch_size, heads, _ = tensor.shape
        expanded = idx.view(batch_size, 1, -1).expand(batch_size, heads, -1)
        return tensor.gather(dim=2, index=expanded)

    return tensor.gather(dim=1, index=idx)
