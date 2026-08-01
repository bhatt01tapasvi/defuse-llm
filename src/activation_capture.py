"""
Phase 3: Capture FFN activations for the pilot prompts, without pruning.

Approved by Naren via WhatsApp, 2026-07-31, after reviewing Phase 2 results
(see progress.md).

For each prompt, captures the input to each layer's MLP down_proj
("mlp.down_proj", shape [num_layers, intermediate_size], fp16):

  1. last_prompt_token      — down_proj input at the final prompt position
                               (the prefill state that produces the first
                               generated token's logits).
  2. mean_prompt_tokens     — down_proj input averaged over all prompt
                               positions (fallback representation).
  3. first_generated_token  — down_proj input when the model processes its
                               own first generated token (one decode step,
                               using the KV cache from prefill).

No full response is generated — this is two forward passes per prompt
(prefill + one decode step), not the 256-token generation used in Phase 2.

Inputs:  datasets/pilot_safety_selectivity/pilot_prompts.jsonl
Outputs: results/pilot_safety_selectivity/activations/<prompt_id>.pt
         results/pilot_safety_selectivity/activation_manifest.jsonl

Resume: safe to re-run — prompt_ids with an existing .pt file are skipped.
"""

import json
import os
import subprocess


def _pick_gpu(preferred_order=(3, 2)):
    """Return a comma-separated CUDA_VISIBLE_DEVICES string.

    Checks GPUs in preferred_order first, then any remaining GPUs, and
    selects the one with the most free memory. Falls back to preferred_order
    as-is if nvidia-smi is unavailable.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        free = {}
        for line in out.strip().splitlines():
            idx, mem = line.split(",")
            free[int(idx.strip())] = int(mem.strip())

        all_gpus = sorted(free.keys())
        ordered = list(preferred_order) + [g for g in all_gpus if g not in preferred_order]
        best = max(ordered, key=lambda g: free.get(g, 0))
        print(f"GPU free memory: { {g: free[g] for g in ordered} }")
        print(f"Selected GPU {best} ({free.get(best, '?')} MiB free)")
        return str(best)
    except Exception as e:
        print(f"nvidia-smi unavailable ({e}), defaulting to GPU {preferred_order[0]}")
        return str(preferred_order[0])


if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = _pick_gpu(preferred_order=(3, 2))

from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
SYSTEM_PROMPT = "You are a helpful assistant."

IN_FILE = Path("datasets/pilot_safety_selectivity/pilot_prompts.jsonl")
OUT_DIR = Path("results/pilot_safety_selectivity")
ACT_DIR = OUT_DIR / "activations"
MANIFEST_FILE = OUT_DIR / "activation_manifest.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_messages(prompt: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


class DownProjCapture:
    """Captures the input to mlp.down_proj at every decoder layer.

    Buffers accumulate in call order: for each prompt, exactly two forward
    calls are made (prefill, then one decode step), so buffers[i][0] is the
    prefill activation and buffers[i][1] is the decode-step activation.
    """

    def __init__(self, model):
        self.buffers = defaultdict(list)
        self.handles = []
        layers = model.model.layers
        for i, layer in enumerate(layers):
            mlp = layer.mlp
            if not hasattr(mlp, "down_proj"):
                raise RuntimeError(f"layer {i} has no mlp.down_proj; unexpected architecture")
            handle = mlp.down_proj.register_forward_pre_hook(self._make_hook(i))
            self.handles.append(handle)

    def _make_hook(self, layer_idx):
        def hook(module, inputs):
            self.buffers[layer_idx].append(inputs[0].detach().to("cpu", torch.float16))
        return hook

    def clear(self):
        self.buffers.clear()

    def remove(self):
        for h in self.handles:
            h.remove()


def load_done_ids(act_dir: Path) -> set:
    if not act_dir.exists():
        return set()
    return {p.stem for p in act_dir.glob("*.pt")}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ACT_DIR.mkdir(parents=True, exist_ok=True)

    prompts = []
    with open(IN_FILE) as f:
        for line in f:
            prompts.append(json.loads(line))
    print(f"Loaded {len(prompts)} prompts from {IN_FILE}")

    done_ids = load_done_ids(ACT_DIR)
    remaining = [p for p in prompts if p["prompt_id"] not in done_ids]
    if done_ids:
        print(f"Resuming: {len(done_ids)} already done, {len(remaining)} remaining")

    if not remaining:
        print("All prompts already captured.")
        return

    print(f"\nLoading model: {MODEL_NAME}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
    cache_dir = os.environ.get("HF_CACHE_DIR", None)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=cache_dir,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    print(f"Model loaded. First param device: {next(model.parameters()).device}")

    num_layers = model.config.num_hidden_layers
    capture = DownProjCapture(model)

    try:
        with open(MANIFEST_FILE, "a") as manifest_f:
            for row in tqdm(remaining, desc="Capturing activations"):
                messages = build_messages(row["prompt"])
                try:
                    input_ids = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    ).to(model.device)
                except Exception:
                    text = f"[INST] {row['prompt']} [/INST]"
                    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

                prompt_len = input_ids.shape[1]

                capture.clear()
                with torch.no_grad():
                    prefill = model(input_ids=input_ids, use_cache=True)
                    next_token_id = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    decode = model(
                        input_ids=next_token_id,
                        past_key_values=prefill.past_key_values,
                        use_cache=True,
                    )
                del decode  # only its hook side-effects are needed

                last_prompt_token = torch.stack(
                    [capture.buffers[i][0][0, -1, :] for i in range(num_layers)]
                )
                mean_prompt_tokens = torch.stack(
                    [capture.buffers[i][0][0].mean(dim=0) for i in range(num_layers)]
                )
                first_generated_token = torch.stack(
                    [capture.buffers[i][1][0, -1, :] for i in range(num_layers)]
                )

                record = {
                    "prompt_id": row["prompt_id"],
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "category": row["category"],
                    "safety_label": row["safety_label"],
                    "model_name": MODEL_NAME,
                    "prompt_len": prompt_len,
                    "first_generated_token_id": int(next_token_id.item()),
                    "last_prompt_token": last_prompt_token,      # [num_layers, intermediate_size], fp16
                    "mean_prompt_tokens": mean_prompt_tokens,    # [num_layers, intermediate_size], fp16
                    "first_generated_token": first_generated_token,  # [num_layers, intermediate_size], fp16
                }
                torch.save(record, ACT_DIR / f"{row['prompt_id']}.pt")

                manifest_f.write(json.dumps({
                    "prompt_id": row["prompt_id"],
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "category": row["category"],
                    "safety_label": row["safety_label"],
                    "prompt_len": prompt_len,
                    "file": str(ACT_DIR / f"{row['prompt_id']}.pt"),
                }) + "\n")
                manifest_f.flush()
    finally:
        capture.remove()

    print(f"\nSaved {len(remaining)} activation files to {ACT_DIR}")
    print(f"Manifest: {MANIFEST_FILE}")


if __name__ == "__main__":
    main()
