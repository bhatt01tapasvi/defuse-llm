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
- Activation files (one per prompt): `results/pilot_safety_selectivity/activations/<prompt_id>.pt`
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

Implementation: `src/activation_capture.py`, self-contained forward-hook
capture (not reusing `src/hook_setup.py`/`src/neuronDefuser.py`, since that
infra is coupled to the separate pruning-focused framework in this repo and
Phase 3 explicitly excludes pruning). Resumable — skips prompt_ids that
already have a `.pt` file.

Verified: 300/300 `.pt` files written, 300/300 manifest rows, activations
distinct across the three capture points (spot-checked via smoke test on 2
prompts before the full run), 789 MB total on disk.

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
