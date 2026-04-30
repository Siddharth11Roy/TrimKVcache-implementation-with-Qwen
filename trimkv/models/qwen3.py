"""TrimKV wrapper around HuggingFace Qwen3.

The wrapper keeps the stock Qwen3 model intact, attaches one retention gate per
decoder layer, and replaces each attention forward method with a TrimKV-aware
version. Only the gates need to be trained.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..attention import retention_weighted_attention
from ..cache_utils import TrimKVCache
from ..retention_gate import RetentionGate


def _training_log_scores(beta: torch.Tensor) -> torch.Tensor:
    """Build full-sequence retention bias with shape (B, H_kv, T_q, T_k)."""
    _, seq_len, _ = beta.shape
    idx = torch.arange(seq_len, device=beta.device)
    age = (idx.view(-1, 1) - idx.view(1, -1)).clamp(min=0).to(beta.dtype)
    causal = idx.view(-1, 1) >= idx.view(1, -1)

    log_beta = beta.clamp(min=1e-8).log().transpose(1, 2)
    scores = log_beta.unsqueeze(2) * age.view(1, 1, seq_len, seq_len)
    return scores.masked_fill(~causal.view(1, 1, seq_len, seq_len), 0.0)


def _fit_attention_mask(
    attention_mask: Optional[torch.Tensor],
    key_len: int,
) -> Optional[torch.Tensor]:
    """Align HuggingFace masks with the shorter key axis after eviction."""
    if attention_mask is None or attention_mask.shape[-1] == key_len:
        return attention_mask
    return attention_mask[..., -key_len:]


def _patched_attention_forward(layer_idx: int):
    """Create a Qwen3 attention `forward` method that uses TrimKV."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[TrimKVCache] = None,
        output_attentions: bool = False,
        use_cache: bool = True,
        cache_position: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        batch_size, seq_len, _ = hidden_states.shape
        num_q_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.head_dim

        q = self.q_proj(hidden_states).view(batch_size, seq_len, num_q_heads, head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        if hasattr(self, "q_norm"):
            q = self.q_norm(q)
        if hasattr(self, "k_norm"):
            k = self.k_norm(k)

        if position_embeddings is not None:
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        beta = self.trimkv_gate(hidden_states)

        if past_key_value is None:
            log_scores = _training_log_scores(beta)
            attn_out = retention_weighted_attention(q, k, v, log_scores, attention_mask)
        else:
            keys, values, _ = past_key_value.update(layer_idx, k, v, beta)
            log_scores = past_key_value.current_log_scores(layer_idx)
            mask = _fit_attention_mask(attention_mask, keys.shape[-2])
            attn_out = retention_weighted_attention(q, keys, values, log_scores, mask)

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, num_q_heads * head_dim)
        return self.o_proj(attn_out), None

    return forward


class TrimKVQwen3ForCausalLM(nn.Module):
    """Thin owner for a patched Qwen3 model and its retention gates."""

    def __init__(self, base_model: nn.Module, memory_size: int, buffer_size: int = 0):
        super().__init__()
        self.base = base_model
        self.config = base_model.config
        self.memory_size = memory_size
        self.buffer_size = buffer_size

        gates = []
        for layer_idx, layer in enumerate(base_model.model.layers):
            attn = layer.self_attn
            ref_weight = attn.q_proj.weight
            gate = RetentionGate(
                hidden_size=self.config.hidden_size,
                num_kv_heads=self.config.num_key_value_heads,
            ).to(device=ref_weight.device, dtype=ref_weight.dtype)
            attn.trimkv_gate = gate
            attn.forward = _patched_attention_forward(layer_idx).__get__(attn, type(attn))
            gates.append(gate)

        self.gates = nn.ModuleList(gates)

    def new_cache(self) -> TrimKVCache:
        return TrimKVCache(
            num_layers=self.config.num_hidden_layers,
            memory_size=self.memory_size,
            buffer_size=self.buffer_size,
        )

    def forward(self, *args, past_key_values: Optional[TrimKVCache] = None, **kwargs):
        return self.base(*args, past_key_values=past_key_values, **kwargs)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        cache = self.new_cache()
        out = self.base(input_ids=input_ids, past_key_values=cache, use_cache=True)
        logits = out.logits[:, -1]
        tokens = [input_ids]

        for _ in range(max_new_tokens):
            if temperature == 0.0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
                next_token = torch.multinomial(probs, 1)

            tokens.append(next_token)
            out = self.base(input_ids=next_token, past_key_values=cache, use_cache=True)
            logits = out.logits[:, -1]

        return torch.cat(tokens, dim=1)

    def gate_parameters(self):
        return self.gates.parameters()


def load_trimkv_qwen3(
    model_name_or_path: str,
    memory_size: int = 512,
    buffer_size: int = 0,
    dtype: torch.dtype = torch.bfloat16,
    device_map: Optional[str] = None,
) -> TrimKVQwen3ForCausalLM:
    from transformers import AutoModelForCausalLM

    kwargs = {"torch_dtype": dtype}
    if device_map is not None:
        kwargs["device_map"] = device_map

    base = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    return TrimKVQwen3ForCausalLM(base, memory_size=memory_size, buffer_size=buffer_size)
