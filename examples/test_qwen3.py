"""End-to-end smoke test: load a Qwen3 model, wrap with TrimKV, generate.

Example:
    python examples/test_qwen3.py \
        --model Qwen/Qwen3-1.7B \
        --memory-size 256 \
        --prompt "Explain why the sky is blue in three sentences."
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer

from trimkv.models.qwen3 import load_trimkv_qwen3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--memory-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=0)
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--gate-ckpt", default=None)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = load_trimkv_qwen3(
        args.model,
        memory_size=args.memory_size,
        buffer_size=args.buffer_size,
        dtype=dtype,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    if args.gate_ckpt is not None:
        state = torch.load(args.gate_ckpt, map_location="cpu")
        for i, g in enumerate(model.gates):
            g.load_state_dict(state[f"layer_{i}"])

    ids = tok(args.prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)
    out = model.generate(ids, max_new_tokens=args.max_new_tokens)
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
