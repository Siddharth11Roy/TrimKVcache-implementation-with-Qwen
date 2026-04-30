"""Gate-only fine-tuning for TrimKV.

The base language model is frozen. Only the retention gates are trained with:

    L = KL(teacher || student) + lambda_cap * L_capacity (+ CE on labels)

The JSONL dataset should contain one object per line with a `"text"` field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trimkv.losses import capacity_loss, distillation_loss
from trimkv.models.qwen3 import TrimKVQwen3ForCausalLM


class JsonlTextDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 1024):
        self.items = [
            json.loads(line)["text"]
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.items:
            raise ValueError(f"no training examples found in {path}")

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        encoded = self.tokenizer(
            self.items[idx],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding="max_length",
        )
        return {key: value.squeeze(0) for key, value in encoded.items()}


def collect_betas(student: TrimKVQwen3ForCausalLM, hidden_states_per_layer):
    """Run each gate on the hidden states entering its layer."""
    return [gate(hidden) for gate, hidden in zip(student.gates, hidden_states_per_layer)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--memory-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lambda-cap", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=0.5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    base_student = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    for param in base_student.parameters():
        param.requires_grad_(False)

    student = TrimKVQwen3ForCausalLM(base_student, memory_size=args.memory_size).to(device)
    for param in student.gate_parameters():
        param.requires_grad_(True)

    optimizer = torch.optim.AdamW(student.gate_parameters(), lr=args.lr)
    dataset = JsonlTextDataset(args.dataset_path, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    step = 0
    while step < args.steps:
        for batch in dataloader:
            step += 1
            batch = {key: value.to(device) for key, value in batch.items()}

            with torch.no_grad():
                teacher_out = teacher(**batch, output_hidden_states=True)

            student_out = student.base(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                output_hidden_states=True,
                past_key_values=None,
                use_cache=False,
            )

            # hidden_states = embedding output + one output per transformer layer.
            # Gate i consumes the input to layer i, so the final state is not used.
            betas = collect_betas(student, student_out.hidden_states[:-1])

            labels = batch["input_ids"].clone()
            labels[batch.get("attention_mask", torch.ones_like(labels)) == 0] = -100

            loss_quality = distillation_loss(
                student_out.logits[:, :-1].contiguous(),
                teacher_out.logits[:, :-1].contiguous(),
                labels=labels[:, 1:].contiguous(),
                ce_weight=args.ce_weight,
            )
            loss_capacity = capacity_loss(betas, memory_size=args.memory_size)
            loss = loss_quality + args.lambda_cap * loss_capacity

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.gate_parameters(), 1.0)
            optimizer.step()

            if step % 10 == 0:
                print(
                    f"step {step:5d}  L={loss.item():.4f}  "
                    f"KL={loss_quality.item():.4f}  cap={loss_capacity.item():.4f}"
                )
            if step >= args.steps:
                break

    out_dir = Path("checkpoints") / f"gates_M{args.memory_size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {f"layer_{idx}": gate.state_dict() for idx, gate in enumerate(student.gates)},
        out_dir / "gates.pt",
    )
    print(f"saved gates to {out_dir / 'gates.pt'}")


if __name__ == "__main__":
    main()
