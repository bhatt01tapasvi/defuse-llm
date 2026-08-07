# Progress — Tapasvi (Phase 1 + Phase 2 + Phase 3; Phase 4 Authorized)

## Status: Phase 1 + Phase 2 + Phase 3 COMPLETE. Phase 4 authorized.

Per `CLAUDE.md`, Tapasvi's scope now includes Phase 4 only. Phase 1-3 are
done and verified on disk. Naren authorized Phase 4 on 2026-08-07; Phase 5+
remain out of scope unless Naren explicitly approves them.

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

## Handoff: Phase 4 Only (Authorized by Naren, 2026-08-07)

Run Phase 4 pattern-separability analysis using the Phase 3 activation
artifacts. Do not modify the capture pipeline or implement any pruning,
interventions, or Phase 5+ work.

1. Evaluate a centroid-similarity detector and a regularized logistic-regression
   linear probe for harmful versus benign prompts.
2. Use the existing train/validation/test splits; do not tune on the test set.
3. Evaluate each activation point and layer. Report AUROC, accuracy, precision,
   recall, and false-positive rate on XSTest.
4. Save reproducible analysis code and the resulting metrics/figures under
   `results/pilot_safety_selectivity/`.

After completing Phase 4, stop and report the results to Naren and Abhishek.
Do not start Phase 5 (static pruning) without Naren's explicit approval.
