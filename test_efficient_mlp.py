"""
Quick test: verify EfficientMLP works end-to-end with LLaMA.
Run on cluster: python test_efficient_mlp.py
"""
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.efficient_mlp import replace_mlps, EfficientLlamaMLP

print("=" * 60)
print("TEST: EfficientMLP with LLaMA")
print("=" * 60)

# Load model on single GPU to avoid multi-GPU device issues
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B", 
    torch_dtype=torch.float16, 
    cache_dir="llm_weights",
    device_map={"": "cuda:0"}  # Force everything to one GPU
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", cache_dir="llm_weights")

device = torch.device("cuda:0")
prompt = "In-memory computing enables parallel processing by performing arithmetic operations directly within memory arrays"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
print(f"Prompt: {len(input_ids[0])} tokens")

# --- Baseline: original model (before replacement) ---
# We already loaded the model, so let's get baseline output first
# Actually, replace_mlps modifies in-place, so we need to run baseline first
print("\n--- Phase 1: Full-path forward (no pruning) ---")
with torch.no_grad():
    out_full = model(input_ids, use_cache=True)
    logits_full = out_full.logits[:, -1, :]
    next_token_full = torch.argmax(logits_full, dim=-1)
    print(f"Next token (full): {next_token_full.item()} = '{tokenizer.decode(next_token_full)}'")

# --- Replace MLPs ---
print("\n--- Phase 2: Replace MLPs with EfficientMLP ---")
mlp_refs = replace_mlps(model, 'llama')
print(f"Replaced {len(mlp_refs)} MLPs")

# Verify full path still produces same output
with torch.no_grad():
    out_after = model(input_ids, use_cache=True)
    logits_after = out_after.logits[:, -1, :]
    next_token_after = torch.argmax(logits_after, dim=-1)
    print(f"Next token (EfficientMLP full path): {next_token_after.item()} = '{tokenizer.decode(next_token_after)}'")
    
    max_diff = (logits_full - logits_after).abs().max().item()
    print(f"Max logit difference (should be ~0): {max_diff:.6e}")
    assert max_diff < 1e-3, f"Full path output mismatch! max_diff={max_diff}"
    print("PASS: Full-path output matches original")

# --- Test reduced path ---
print("\n--- Phase 3: Test reduced-weight path (50% pruning) ---")
intermediate_size = model.config.intermediate_size if model.config.intermediate_size else 4 * model.config.hidden_size
k = intermediate_size // 2  # Keep 50%

for layer_name, mlp in mlp_refs.items():
    # Random indices for testing
    kept_indices = torch.randperm(intermediate_size, device=device)[:k].sort()[0]
    mlp.prepare_reduced_weights(kept_indices)

print(f"Prepared reduced weights: {k}/{intermediate_size} neurons per layer")

# Run with reduced weights
with torch.no_grad():
    out_reduced = model(input_ids, use_cache=True)
    logits_reduced = out_reduced.logits[:, -1, :]
    next_token_reduced = torch.argmax(logits_reduced, dim=-1)
    print(f"Next token (reduced 50%): {next_token_reduced.item()} = '{tokenizer.decode(next_token_reduced)}'")
    
    # Should differ from full (we pruned randomly!)
    diff = (logits_full - logits_reduced).abs().max().item()
    print(f"Max logit difference from full (expected non-zero): {diff:.4f}")

# --- Test clear and revert ---
print("\n--- Phase 4: Clear reduced weights, verify revert to full ---")
for mlp in mlp_refs.values():
    mlp.clear_reduced_weights()

with torch.no_grad():
    out_reverted = model(input_ids, use_cache=True)
    logits_reverted = out_reverted.logits[:, -1, :]
    
    max_diff_revert = (logits_full - logits_reverted).abs().max().item()
    print(f"Max logit difference after revert (should be ~0): {max_diff_revert:.6e}")
    assert max_diff_revert < 1e-3, f"Revert output mismatch! max_diff={max_diff_revert}"
    print("PASS: Reverted output matches original")

# --- Timing comparison ---
print("\n--- Phase 5: Generation speed comparison ---")
num_tokens = 1000

# Full path timing
for mlp in mlp_refs.values():
    mlp.clear_reduced_weights()

with torch.no_grad():
    past = model(input_ids, use_cache=True).past_key_values
    cur_ids = input_ids
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_tokens):
        out = model(cur_ids[:, -1:], past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        cur_ids = torch.cat([cur_ids, next_tok], dim=-1)
    torch.cuda.synchronize()
    t_full = time.perf_counter() - t0

full_text = tokenizer.decode(cur_ids[0])
print(f"Full path: {num_tokens} tokens in {t_full*1000:.1f} ms ({t_full/num_tokens*1000:.2f} ms/token)")

# Reduced path timing (50% pruning)
for layer_name, mlp in mlp_refs.items():
    kept_indices = torch.randperm(intermediate_size, device=device)[:k].sort()[0]
    mlp.prepare_reduced_weights(kept_indices)

with torch.no_grad():
    past = model(input_ids, use_cache=True).past_key_values
    cur_ids = input_ids
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_tokens):
        out = model(cur_ids[:, -1:], past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        cur_ids = torch.cat([cur_ids, next_tok], dim=-1)
    torch.cuda.synchronize()
    t_reduced = time.perf_counter() - t0

reduced_text = tokenizer.decode(cur_ids[0])
print(f"Reduced path (50%): {num_tokens} tokens in {t_reduced*1000:.1f} ms ({t_reduced/num_tokens*1000:.2f} ms/token)")
print(f"Speedup: {t_full/t_reduced:.2f}x")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
