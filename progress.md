# Progress — Tapasvi (Phase 1 + Phase 2 + Phase 3)

## Status: Phase 1 + Phase 2 + Phase 3 COMPLETE. Holding at scope boundary.

Per `CLAUDE.md`, Tapasvi's scope is Phase 1, Phase 2, and Phase 3 only
(Phase 3 approved by Naren via WhatsApp, 2026-07-31, after reviewing Phase 2
results below). All three are done and verified on disk. No Phase 4 work
(linear probes / separability analysis) has been started, per the explicit
scope boundary in `CLAUDE.md`.

## Files

- Prompt file: `datasets/pilot_safety_selectivity/pilot_prompts.jsonl`
- Dataset stats: `datasets/pilot_safety_selectivity/dataset_stats.json`
- Dense baseline output: `results/pilot_safety_selectivity/dense_baseline_generations.jsonl`
- Metrics summary: `results/pilot_safety_selectivity/dense_baseline_metrics.json`
- Activation files (one per prompt): `results/pilot_safety_selectivity/activations/individual/<prompt_id>.pt`
- Activation aggregate (all prompts stacked, for Phase 4): `results/pilot_safety_selectivity/activations/pilot_activations_aggregated.pt`
- Activation manifest: `results/pilot_safety_selectivity/activation_manifest.jsonl`

## Dense Baseline Metrics (Mistral-7B-Instruct-v0.3)

| Metric | Value |
|--------|-------|
| HarmBench harmful compliance rate | 0.68 |
| HarmBench refusal rate | 0.32 |
| JailbreakBench benign refusal rate | 0.00 |
| XSTest over-refusal rate | 0.04 |
| HarmBench avg output length (words) | 164.8 |
| JailbreakBench benign avg output length | 165.1 |
| XSTest avg output length | 121.4 |

Full report with representative examples and issues encountered:
`research_planner/NeurIPS Pattern-Triggered Pruning Plan.md`, section
"Tapasvi: Phase 1 + Phase 2 Report (2026-04-22)".

## Phase 3: Activation Collection

Captured, per pilot prompt (300 total: 100 harmbench / 100 jailbreakbench_benign
/ 100 xstest — 100 harmful / 200 benign by safety label), the input to every
decoder layer's `mlp.down_proj` (32 layers × 14336-dim, fp16) at three points:

1. `last_prompt_token` — down_proj input at the final prompt position (prefill).
2. `mean_prompt_tokens` — down_proj input averaged over all prompt positions.
3. `first_generated_token` — down_proj input when the model processes its own
   first generated token (one decode step from the prefill KV cache).

No pruning and no full-response generation were done — just prefill + one
decode step per prompt, matching the Phase 3 spec in the plan doc exactly
(`research_planner/NeurIPS Pattern-Triggered Pruning Plan.md:616-629`).

Implementation: `src/collect_activations.py`. Follows your Phase 3
instructions (`ea2e1fc`, picked up late — `origin` was unreachable when
Tapasvi started this work, see note below) on script naming and dual
storage (individual `.pt` per prompt under `activations/individual/`, plus
one aggregated `activations/pilot_activations_aggregated.pt` with all 300
prompts stacked per activation type for fast Phase 4 loading).

One deliberate deviation from those instructions: activation capture uses
self-contained forward hooks, not `src.hook_setup.setup_hooks_mistral()` +
`NeuronDefuser`. Reason: that hook's down_proj pre-hook unconditionally
calls `neuronDefuser.defuse_neurons()` on the tensor that continues through
the forward pass (`src/hook_setup.py:507-517`, "CRITICAL: For mlp_down, we
MUST defuse neurons"). `defuse_neurons()` (`src/neuronDefuser.py:195`) is a
stateful masking/pruning function needing precomputed `forward_proxies_max`/
`forward_proxies_mean` and EMA state — getting it into a verified no-op
"dense" mode isn't a flag flip, and if misconfigured it would silently prune
the very activations Phase 3 needs to capture *without* pruning (the plan
doc's own Phase 3 goal). Tapasvi wrote a minimal hook that only reads
`down_proj`'s input and does not touch the forward pass, to avoid that risk.
Happy to switch to the shared hook infra if you can point to the exact
`NeuronDefuser` init args that make `defuse_neurons()` a guaranteed no-op.

Resumable — skips prompt_ids that already have an individual `.pt` file;
the aggregate is rebuilt from whatever's on disk each run.

Verified: 300/300 individual `.pt` files, 300/300 manifest rows, aggregate
file shape `[300, 32, 14336]` per activation type (matches prompt count ×
layer count × intermediate size), activations distinct across the three
capture points (spot-checked via smoke test on 2 prompts before the full
run). 789 MB individual + 788 MB aggregate on disk.

### Note on `ea2e1fc`

This commit (your Phase 3 instructions doc update, pushed to `origin`
several days ago) wasn't visible to Tapasvi until today — `origin`
(`ATygah/defuse-llm`) was returning "repository not found" when this
session started, and the cached `origin/naren_research` ref predated your
push. Once access came back, this reconciliation was done in the same
session. Worth checking whether `origin` access dropped for others too.

## Question for Naren

Phase 3 activation capture is done and sitting on disk, unexamined —
Tapasvi has not looked at separability, run any probes, or touched pruning,
per the Phase 4+ scope block in `CLAUDE.md`.

**Naren — how should Tapasvi proceed?**

- Cleared to start Phase 4 (pattern separability: centroid similarity +
  logistic regression probe, per-layer AUROC/accuracy/precision/recall/FPR)?
- Hold and wait for you + Abhishek to review the captured activations first?
- Anything to change about the Phase 3 capture (which activation points,
  which layers, fp16 vs fp32, prompt-only vs including a longer generation
  window) before Phase 4 work begins?
