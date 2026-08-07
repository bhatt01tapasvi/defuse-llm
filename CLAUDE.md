# defuse-llm — Claude Project Context

## Project

NeurIPS research: pattern-triggered FFN pruning for LLM safety.
Goal: detect harmful activation patterns during inference, conditionally prune FFN neurons
to reduce harmful compliance while preserving benign helpfulness.

Repo: https://github.com/ATygah/defuse-llm
Branch: naren_research

## Environment

Always use the `tapasvi-env` conda environment:

```bash
conda run -n tapasvi-env python <script>
# or
conda activate tapasvi-env && python <script>
```

## Scope Boundary — NEVER go beyond Phase 4

Tapasvi owns Phase 1, Phase 2, Phase 3, and Phase 4 only.

Phase 1: dataset setup → `datasets/pilot_safety_selectivity/pilot_prompts.jsonl`
Phase 2: dense baseline → `results/pilot_safety_selectivity/dense_baseline_generations.jsonl`
Phase 3: activation collection (approved by Naren via WhatsApp, 2026-07-31, after
reviewing Phase 2 results — see `progress.md`)
Phase 4: pattern separability analysis (approved by Naren, 2026-08-07)

DO NOT implement:
- Static or pattern-triggered pruning (Phase 5-6)
- Causal experiments (Phase 7)

After Phase 4, stop and report results to Naren and Abhishek. Do not begin
Phase 5 unless Naren explicitly approves it.

## Key Files

| File | Purpose |
|------|---------|
| `src/pilot_dataset.py` | Phase 1: build pilot prompt JSONL |
| `src/dense_baseline.py` | Phase 2: run dense model, compute metrics |
| `src/collect_activations.py` | Phase 3: capture FFN down_proj activations per prompt |
| `configs/defuse_experiment_config.yaml` | Model config (Mistral path) |
| `research_planner/NeurIPS Pattern-Triggered Pruning Plan.md` | Full research plan |

## Output Paths

```
datasets/pilot_safety_selectivity/pilot_prompts.jsonl
datasets/pilot_safety_selectivity/dataset_stats.json
results/pilot_safety_selectivity/dense_baseline_generations.jsonl
results/pilot_safety_selectivity/dense_baseline_metrics.json
results/pilot_safety_selectivity/activations/individual/<prompt_id>.pt
results/pilot_safety_selectivity/activations/pilot_activations_aggregated.pt
results/pilot_safety_selectivity/activation_manifest.jsonl
```

## Model

mistralai/Mistral-7B-Instruct-v0.3

## GPU Preference

4× NVIDIA RTX A5000 (GPU 0-3). Priority order: **GPU 3 → GPU 2 → others**.

Scripts auto-select using `_pick_gpu(preferred_order=(3, 2))` — checks free memory via
`nvidia-smi` and picks the GPU with the most free memory, checking GPU 3 and GPU 2 first
(used as tiebreak), then any remaining GPU.

To override manually:
```bash
CUDA_VISIBLE_DEVICES=3 conda run -n tapasvi-env python <script>
```

## Working Directory

Always stay inside /usr1/home/s124mdg41_05/Tapas/defuse-llm.
Never access paths outside this directory.
