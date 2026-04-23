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

## Scope Boundary — NEVER go beyond Phase 2

Tapasvi owns Phase 1 and Phase 2 only.

Phase 1: dataset setup → `datasets/pilot_safety_selectivity/pilot_prompts.jsonl`
Phase 2: dense baseline → `results/pilot_safety_selectivity/dense_baseline_generations.jsonl`

DO NOT implement:
- Activation capture or hooks (Phase 3)
- Linear probes (Phase 4)
- Static or pattern-triggered pruning (Phase 5-6)
- Causal experiments (Phase 7)

After Phase 2, report results to Naren and Abhishek.

## Key Files

| File | Purpose |
|------|---------|
| `src/pilot_dataset.py` | Phase 1: build pilot prompt JSONL |
| `src/dense_baseline.py` | Phase 2: run dense model, compute metrics |
| `configs/defuse_experiment_config.yaml` | Model config (Mistral path) |
| `research_planner/NeurIPS Pattern-Triggered Pruning Plan.md` | Full research plan |

## Output Paths

```
datasets/pilot_safety_selectivity/pilot_prompts.jsonl
datasets/pilot_safety_selectivity/dataset_stats.json
results/pilot_safety_selectivity/dense_baseline_generations.jsonl
results/pilot_safety_selectivity/dense_baseline_metrics.json
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
