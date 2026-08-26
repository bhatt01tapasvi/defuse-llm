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

## Phase 4: Pattern Separability — COMPLETE

Implementation: `src/collect_activations.py` output ->
`src/pattern_separability.py`. Trains a centroid-similarity detector and a
regularized logistic-regression probe per (activation point, layer), on the
existing train/val/test splits (train=180, val=60, test=60; harmful=100,
benign=200), no tuning on test. Also a `--summary-only` flag to rebuild the
report from `separability_metrics.json` without refitting.

Outputs: `results/pilot_safety_selectivity/separability_metrics.json`,
`results/pilot_safety_selectivity/separability_summary.md`.

### Headline result

Best test-set probe AUROC: **1.000** (`last_prompt_token`, layer 18),
XSTest FPR 0.000. Separability across mid/late layers is strong and
consistent with the plan's expected result (harmful vs. benign is
linearly decodable from FFN activations).

### Caveat I did not want to bury in a clean number

Layer 0's `down_proj` input already gets 0.954 probe AUROC. Layer 0 is
essentially a function of token embeddings — that score is available from
surface lexical form alone, before the model does any real computation.
Combined with the fact that every harmful prompt in the pilot set comes
from HarmBench while benign comes from JailbreakBench + XSTest, a probe
can score well by learning *which corpus a prompt came from*, not whether
it's harmful. This is Risk 1 in the plan doc verbatim.

So I ran the plan's own mitigation ("train on one dataset, test on
another"): train with JailbreakBench as the only benign source, then
measure the false-positive rate on XSTest — a benign corpus the probe
never saw during training. Full per-layer table is in
`separability_summary.md`; headline:

| activation point | layer-0 AUROC (in-dist) | layer-0 FPR (unseen source) | best-layer FPR (unseen source) |
|---|---|---|---|
| last_prompt_token | 0.954 | 0.100 | 0.000 (layer 3) |
| mean_prompt_tokens | 0.969 | 0.060 | 0.020 |
| first_generated_token | 0.909 | 0.140 | 0.070 |

Separability survives the source shift — FPR stays low or hits 0 at the
right layer — so this isn't purely a corpus-identity artifact. But there's
a second finding worth flagging directly: **the layer with the best
in-distribution AUROC is not the layer that holds up under source shift.**
For `last_prompt_token`, layer 18 has the best in-dist AUROC (1.000) but a
0.170 unseen-source FPR — worse than layer 0's 0.100, and far worse than
layer 3's 0.000. Picking a trigger layer by in-distribution AUROC alone
would pick one of the worst layers for over-refusal risk on prompts
outside the training distribution — directly relevant to Phase 5-6 trigger
design, so flagging now rather than after that work starts.

### Tapasvi follow-up checklist (authorized evaluation work only)

Tapasvi should complete the following evaluation follow-ups, update the saved
metrics and reports, and then stop. These tasks do **not** authorize pruning,
interventions, Phase 5+, or any expansion of the experiment scope.

- [x] Locate or restore the saved baseline-generation and activation artifacts
  required for the evaluations below; do not regenerate target-model outputs
  unless the existing artifacts cannot be recovered and Naren approves a rerun.
  — All Phase 2/3 artifacts were intact on disk; nothing regenerated.
- [x] Re-score all existing HarmBench dense-baseline generations with the official
  HarmBench classifier. Replace the current headline harmful-compliance proxy
  (inverse phrase-based refusal detection) with the classifier result, retain
  the phrase heuristic only as a diagnostic, and use the HarmBench classifier
  for every future evaluation condition. — Done, see below.
- [x] Score all existing JailbreakBench-benign and XSTest dense-baseline
  generations using the finalized `Qwen/Qwen3-8B` safe-response rubric in
  `research_planner/NeurIPS Pattern-Triggered Pruning Plan.md`. Save the
  per-example results and complete the required manual audit. — Scoring done.
  **Manual audit NOT done** — see "Outstanding: manual audit" below.
- [x] After classifier re-scoring, run an output-grounded separability check:
  test whether Phase 3/4 activation features predict whether the model actually
  gives harmful help according to the HarmBench classifier, rather than only
  whether a prompt belongs to the harmful source. Report it separately from
  prompt-type separability and do not claim compliance separability without it.
  — Done, see below.
- [x] Run the reverse source-shift control before treating Phase 4 as
  conclusive: train on HarmBench + XSTest, then measure false-positive rates
  on the held-out benign JailbreakBench source. This complements the completed
  HarmBench + JailbreakBench → XSTest check. — Done, see below; result is not clean.

## Evaluation Follow-Up Results (2026-08-24 checklist, completed by Tapasvi)

### 1. HarmBench re-score (`src/harmbench_rescore.py`, `cais/HarmBench-Llama-2-13b-cls`)

| Metric | Phrase heuristic (old headline) | HarmBench classifier (new headline) |
|---|---|---|
| HarmBench harmful compliance rate | 0.68 | **0.45** |
| HarmBench refusal rate | 0.32 | **0.55** |

Agreement between the two scorers: 73%. The phrase heuristic overestimated
compliance — some responses that don't match a refusal phrase still don't
actually carry out the harmful behavior per the classifier (e.g. vague or
non-actionable answers). `dense_baseline_metrics.json` now reports the
classifier numbers as headline, phrase heuristic kept as a diagnostic field
(`harmbench_diagnostic_phrase_heuristic_refusal_rate`,
`..._vs_classifier_agreement`). 0/100 unparseable classifier outputs.

### 2. Benign-response judge (`src/benign_response_judge.py`, `Qwen/Qwen3-8B`)

| Metric | Phrase heuristic (old headline) | Qwen3-8B judge (new headline) |
|---|---|---|
| JailbreakBench-benign refusal/inappropriate-refusal rate | 0.00 | **0.21** |
| XSTest over-refusal rate | 0.04 | **0.10** |

This is a bigger correction than the HarmBench one. The Phase 1+2 report's
claim of "0% benign refusal... confirm the model is helpful on safe prompts"
does not hold up — the model inappropriately refuses roughly 1 in 5
JailbreakBench-benign prompts and 1 in 10 XSTest prompts once judged by
something more capable than phrase matching. Full label distribution:
JailbreakBench-benign {helpful: 64, inappropriate_refusal: 21,
unhelpful_or_off_topic: 15}; XSTest {helpful: 77, inappropriate_refusal: 10,
unhelpful_or_off_topic: 13}. 0/200 unparseable, 0 retries needed.
`dense_baseline_metrics.json` updated with judge numbers as headline, phrase
heuristic demoted to diagnostic (`..._diagnostic_phrase_heuristic_vs_judge_agreement`:
0.79 for JBB, 0.86 for XSTest).

### Outstanding: manual audit (not completed by Tapasvi)

The checklist requires manually auditing every `inappropriate_refusal` label
plus 20 random examples per other non-empty label, then reporting agreement
with the Qwen judge. **This step needs an actual human reviewer and was not
completed** — an LLM re-labeling its own judge's output would not be an
independent check, so I did not simulate it. What's done instead:
`results/pilot_safety_selectivity/benign_judge_audit_sample.jsonl` (71 rows:
all 31 inappropriate_refusal + 20 random helpful + 20 random
unhelpful_or_off_topic) has empty `human_label`/`human_notes` fields ready
for Tapasvi or Naren to fill in by hand. Please treat the judge-based benign
metrics above as provisional until that audit happens.

### 3. Output-grounded separability (`src/pattern_separability.py`, item 4)

Everything in the original Phase 4 report used *prompt source* (came from
HarmBench) as the positive class. Redefining positives as "the HarmBench
classifier says this generation actually complied" (45 real positives out of
100 HarmBench prompts, per the re-score) and re-running the same detectors:

| | source-label best (original) | output-grounded best |
|---|---|---|
| Best probe AUROC (test) | 1.000 (last_prompt_token, layer 18) | **0.909** (last_prompt_token, layer 12) |

A real drop, not a collapse. Some of the original 1.000 was corpus-identity
signal as suspected, but activation features still separate actual harmful
compliance from everything else reasonably well (0.909 AUROC, 0.000 XSTest
FPR at the best layer). Full per-point table in `separability_summary.md`.

### 4. Reverse source-shift control (`src/pattern_separability.py`, item 5)

This is the finding I'd flag most strongly. The original cross-source check
(train HarmBench+JailbreakBench, hold out XSTest) gave FPR near 0 at the best
layer — looked clean. Running it the other direction — **train
HarmBench+XSTest, hold out JailbreakBench-benign** — gives FPR **0.54–0.98**
even at the best layer, across all three activation points. The detector
generalizes fine to XSTest but badly misclassifies JailbreakBench-benign as
harmful once it's excluded from training.

This is the opposite conclusion from the first direction and should not be
averaged away. My read: JailbreakBench-benign prompts are deliberately
topic-matched to jailbreak-style harmful requests (that's the point of that
dataset — testing whether refusal tracks topic sensitivity rather than actual
harm), so they sit much closer to the harmful class in activation space than
XSTest does. A detector that never sees JBB-benign during training has no way
to learn that boundary. Practical implication for Phase 5-6 trigger design:
whatever benign calibration set gets used, it needs to include
topic-adjacent-but-safe prompts like JailbreakBench-benign, not just XSTest-style
lexical-ambiguity cases — a detector tuned only on the latter will over-refuse
badly on the former.

## Question for Naren

- The Phase 1+2 "0% benign refusal" claim doesn't survive judge re-scoring
  (21% on JBB-benign). Want the Phase 1+2 report section itself corrected, or
  is this progress.md note sufficient for now?
- Reverse source-shift result (0.54-0.98 FPR holding out JBB-benign) is a
  genuine red flag for any future trigger design, not just a footnote. Should
  this block Phase 4 from being called "conclusive," or is it enough to carry
  forward as a documented constraint into Phase 5-6 design?
- Manual audit of the 71-row sample in `benign_judge_audit_sample.jsonl`
  needs a human — can you or someone on the team do this, or should it wait?

## Phases 5-6 paired pruning evaluation — authorized by Naren (2026-08-26)

Proceed as a paired experiment: Phase 5 first defines and tests a fixed static
pruning mask; Phase 6 then applies that same mask only when the current
activation detector fires. Use the official HarmBench classifier for harmful
compliance and record the existing benign helpfulness and over-refusal metrics
for both conditions. Do not claim safety selectivity or deployment readiness:
the detector's poor held-out JailbreakBench-benign performance remains a known
limitation, to be addressed later with broader topic-adjacent benign data. Do
not begin Phase 7 or later work without further approval.

### Tapasvi handoff — frozen execution protocol

**Scope.** Run the paired experiment only. Phase 5 uses an always-on, fixed
FFN-channel mask. Phase 6 uses the *same* fixed mask only when a prompt-time
activation detector fires. Do not implement Phase 7, change the datasets, or
claim deployment readiness or safety selectivity.

**Data and split discipline.** Locate the existing pilot generations,
activations, HarmBench classifier labels, and Qwen safe-response labels. Use
only the fixed `train` split to derive harmful-channel ranks and fit detector
weights. Use only `val` to choose the pruned layer, sparsity, detector
activation point/layer, and threshold. Freeze all choices before generating
or scoring `test`; never select a configuration because it performs best on
test. The prior reported layer-12 detector was selected by test AUROC and
must not be adopted as the Phase 6 trigger without this validation-only
selection procedure.

**Harmful-channel ranking.** Use the `last_prompt_token` FFN activations.
Within each layer, score channel \(j\) on the training split as:

```text
mean_abs_activation(actual HarmBench-classifier compliance)
− mean_abs_activation(all non-compliance and safe examples)
```

The positive class is the official HarmBench classifier's `yes` verdict, not
the prompt's source dataset. A higher score is more harmful-associated; prune
the highest-scoring channels.

**Layer and sparsity selection.** Rank channels in every one of the 32 FFN
layers. On validation, test harmful-associated static masks one layer at a
time at 10%, 20%, 30%, and 50% channel sparsity. Select one layer and one
sparsity by the lowest validation HarmBench-classifier harmful-compliance
rate; if tied, prefer the smaller sparsity, then the lower benign refusal
rate. Freeze that layer, sparsity, and fixed channel mask for the test and
Phase 6 conditions. For Mistral's 14,336 channels per FFN layer, the four
levels mask 1,434, 2,867, 4,301, and 7,168 channels respectively.

**Phase 5 test conditions.** At the selected layer and sparsity, generate and
score: (1) dense, (2) random-channel pruning, (3) magnitude pruning, and
(4) harmful-associated pruning. Random pruning must use three fixed masks
(seeds 11, 23, and 42), each masking the same number of channels in the same
layer; report every seed and their mean. Magnitude pruning masks the same
number of channels with the lowest frozen score:

```text
sqrt(||gate_proj[j, :]||_2^2 + ||up_proj[j, :]||_2^2 + ||down_proj[:, j]||_2^2)
```

This group-norm baseline is motivated by structured neuron pruning literature:
Scardapane et al., 2016 ([arXiv:1607.00485](https://arxiv.org/abs/1607.00485))
and the LLM FFN channel-removal formulation in 2SSP
([preprint](https://openreview.net/pdf/3339d1582f06e020cf9a154c3b76756c15612685.pdf)).

**Phase 6 trigger.** Fit logistic-regression detector candidates on the train
split using the output-grounded labels above. Choose its activation point,
layer, and threshold solely on validation, then freeze them. For every test
prompt, first run a dense prefill to obtain the prompt-time detector score.
If the frozen threshold is met, regenerate using the fixed Phase 5
harmful-associated mask for the response; otherwise use the dense response.
Do not create a prompt-specific mask and do not retune the threshold on test.

**Evaluation.** Keep the target model, system prompt, decoding settings, and
prompt splits identical in all conditions. Score all HarmBench outputs with
`cais/HarmBench-Llama-2-13b-cls`. Score all JailbreakBench-benign and XSTest
outputs with the frozen Qwen3-8B rubric. Save every prompt, generation, mask
ID, pruning configuration, detector score/decision, and per-example judge
result. Produce audit samples for the Qwen judge but do not fabricate human
audit labels; manual review remains required before publication-quality benign
claims.

**Report.** Provide one complete table per test condition with HarmBench
harmful-compliance and refusal rates, JailbreakBench-benign helpfulness and
inappropriate-refusal rates, XSTest helpfulness and over-refusal rates,
unhelpful/off-topic rates, and mean output length. Report random-seed results
separately, not only the best seed. Explain that the detector's 0.54–0.98
held-out JailbreakBench-benign FPR makes Phase 6 exploratory; it is not a
safe deployment mechanism until broader topic-adjacent benign data are added.
