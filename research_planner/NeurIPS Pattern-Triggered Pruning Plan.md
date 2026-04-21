# NeurIPS Plan: Pattern-Triggered Pruning for Harmful Computation

## One-Sentence Idea

Harmful behavior in an LLM is not stored in fixed "bad neurons"; it appears as a temporary activation pattern, and we can detect that pattern during inference and selectively prune FFN units to disrupt harmful compliance while preserving benign helpfulness.

## Core Claim

The paper should argue:

> Harmful compliance in aligned LLMs has detectable activation signatures and is more fragile than benign helpfulness under structured FFN perturbation. We exploit this with pattern-triggered pruning at inference time.

This is stronger than saying "we found harmful neurons." The contribution is about conditional activation patterns, online detection, and causal disruption.

## Why This Can Be NeurIPS-Worthy

The work becomes a serious NeurIPS paper if it combines:

1. A new method: pattern-triggered inference-time FFN pruning.
2. A scientific claim: harmful behavior is context-dependent and more fragile than benign behavior.
3. Causal evidence: pruning detected harmful activation patterns reduces harmful compliance.
4. Strong evaluation: harmful compliance drops more than benign helpfulness, across multiple datasets and models.

## Proposed Method

### 1. Offline Pattern Extraction

Run harmful and benign prompts through the model and collect FFN activations layer by layer.

For each monitored layer:

```text
harmful_signature = mean(harmful_activations) - mean(benign_activations)
```

Also store:

- top-k neurons frequently active on harmful prompts
- top-k harmful co-activation patterns
- layer-wise harmful-vs-benign separability scores

### 2. Online Detection

During inference, monitor the current activation vector at selected layers.

Simple detector:

```text
harmful_score = cosine(current_activation, harmful_signature)
```

Trigger pruning when:

```text
harmful_score > threshold
```

More robust detector:

```text
trigger if at least m out of n monitored layers exceed threshold
```

Example:

```text
trigger if 3 out of 5 selected layers look harmful
```

### 3. Conditional Pruning

If the harmful activation pattern is detected:

- prune FFN neurons most associated with the harmful signature
- apply pruning only for the current generation
- avoid pruning on normal benign prompts

This makes the intervention conditional, not always-on.

## What Patterns To Detect

Start with simple, defensible patterns before moving to complex interpretability tools.

### Pattern Type 1: Layer-Wise Harmful Direction

Compute a direction in activation space:

```text
harmful_direction = mean(harmful) - mean(benign)
```

This is aligned with representation engineering and refusal-direction work.

### Pattern Type 2: Top-k Harmful FFN Neurons

Find neurons that are much more active on harmful prompts than matched benign prompts.

Score:

```text
neuron_score = avg_activation_harmful - avg_activation_benign
```

Prune the highest-scoring neurons only when the harmful pattern is detected.

### Pattern Type 3: Co-Activation Patterns

Instead of treating neurons independently, track sets of neurons that light up together during harmful prompts.

This supports the main claim:

> Harmfulness is a pattern of computation, not a property of one isolated neuron.

### Pattern Type 4: Linear Probe Detector

Train a simple classifier:

```text
input: activation vector from selected layers
output: harmful-pattern probability
```

Use logistic regression or a small MLP. Logistic regression is preferred first because it is easier to interpret and defend.

### Pattern Type 5: Sparse Autoencoder Features

Optional later-stage analysis.

Use sparse autoencoders to decompose activations into interpretable features, then check whether harmful prompts activate recurring safety-relevant features.

Do not start here unless the simpler activation-signature approach is insufficient.

## Standard Methods To Build On

Relevant established methods:

- Representation engineering: learning high-level concept directions from activation differences.
- Refusal direction analysis: identifying directions in activation space related to refusal behavior.
- Linear probing: checking whether harmful-vs-benign state is linearly decodable.
- Activation patching: replacing or modifying activations to test whether they causally affect model behavior.
- Sparse autoencoders: decomposing activations into more interpretable features.

Use these as baselines and interpretability support, but keep the main method focused on pattern-triggered pruning.

## Datasets

Use standard datasets. Keep training/calibration prompts separate from final test prompts.

### Harmful Direct Prompts

Use:

1. HarmBench
   - Standardized red-teaming framework.
   - Includes harmful behaviors and automated evaluation tooling.
   - Link: https://github.com/centerforaisafety/HarmBench

2. SORRY-Bench
   - Fine-grained unsafe-refusal benchmark.
   - Balanced safety taxonomy across many unsafe categories.
   - Link: https://sorry-bench.github.io/

### Jailbreak Prompts

Use:

1. JailbreakBench
   - NeurIPS 2024 Datasets and Benchmarks benchmark.
   - Has harmful and benign behaviors plus jailbreak evaluation tooling.
   - Link: https://github.com/JailbreakBench/jailbreakbench

2. StrongREJECT
   - Jailbreak evaluation benchmark with harmful prompts and evaluators.
   - Link: https://strong-reject.readthedocs.io/en/latest/

### Benign and Over-Refusal Prompts

Use:

1. JailbreakBench benign set
   - Useful for directly matched harmful-vs-benign evaluation.

2. XSTest
   - Safe prompts that look superficially unsafe.
   - Good for measuring false refusal / exaggerated safety.
   - Link: https://huggingface.co/datasets/Paul/XSTest

3. OR-Bench
   - Large-scale over-refusal benchmark.
   - Use the hard subset for efficient evaluation.
   - Link: https://arxiv.org/abs/2405.20947

### Judge Calibration

Use:

1. Aegis 2.0
   - Human-annotated content safety dataset with broad hazard taxonomy.
   - Useful for calibrating or validating moderation-style judges.
   - Link: https://arxiv.org/abs/2501.09004

## Recommended Dataset Mix

For a strong first paper:

- Direct harmful: HarmBench + SORRY-Bench
- Jailbreak: JailbreakBench + StrongREJECT
- Benign / over-refusal: JailbreakBench benign + XSTest + OR-Bench hard subset
- Judge calibration: Aegis 2.0

## Pilot Dataset Selection

For the immediate pilot, do not use all datasets. Use a small, balanced sample that can quickly tell us whether the idea has signal.

Use exactly:

1. HarmBench harmful direct: 100 prompts.
2. JailbreakBench benign: 100 prompts.
3. XSTest benign-sensitive: 100 prompts.
4. JailbreakBench harmful wrapped: 100 prompts, optional after the first dense baseline is working.

The first three sets are enough for Phase 1 and Phase 2.

### HarmBench: 100 Harmful Direct Prompts

Do not sample randomly. Use stratified sampling across harm categories.

Target distribution:

```text
10 cyber / malware
10 fraud / scams
10 weapons / explosives
10 chemical / biological
10 violence / physical harm
10 self-harm
10 privacy / doxxing
10 misinformation / manipulation
10 illegal activity
10 hate / harassment / abuse
```

If HarmBench category names differ, map them into these broad groups and sample evenly. Use a fixed seed and save the selected prompt IDs.

### JailbreakBench Benign: 100 Safe Prompts

Select benign prompts that roughly match the harmful prompt domains.

Target distribution:

```text
10 benign cybersecurity
10 benign chemistry
10 benign biology
10 benign legal / policy
10 benign finance
10 benign health
10 benign privacy / data protection
10 benign history / politics
10 benign writing / persuasion
10 benign general technical help
```

The comparison should not be harmful chemistry versus harmless trivia. Match domain and reasoning load where possible.

### XSTest: 100 Benign-Sensitive Prompts

Use XSTest to measure over-refusal. Select prompts that look superficially unsafe but are actually safe.

Target distribution:

```text
safe figurative language
safe historical discussion
safe fictional violence
safe educational chemistry / biology
safe cybersecurity defense
safe medical information
safe legal information
safe self-harm prevention / support
safe weapons policy / history
safe privacy / security best practices
```

The goal is to test whether the detector or pruning method overreacts to sensitive-but-safe prompts.

### JailbreakBench Harmful Wrapped: 100 Prompts

Use this after the first dense baseline works.

Preferred structure:

```text
50 direct harmful behaviors
the same 50 behaviors with jailbreak wrappers
```

This gives a clean direct-vs-jailbreak comparison for the same underlying intent.

### Pilot Split

For each 100-prompt set:

```text
60 train
20 validation
20 test
```

Use:

```text
train: learn activation signatures / train probes
validation: choose detection thresholds and pruning sparsity
test: final pilot result
```

Do not tune thresholds, sampling rules, or pruning settings on the test split.

## Metrics

### Safety Metrics

1. Harmful Compliance Rate
   - Fraction of harmful prompts where the model gives actionable unsafe help.

2. Refusal Rate
   - Fraction of harmful prompts where the model refuses.

3. Partial Compliance Severity
   - Graded score for vague, partial, or actionable harmful help.

4. Jailbreak Attack Success Rate
   - Harmful compliance rate under jailbreak wrappers.

### Utility Metrics

1. Benign Helpfulness
   - How well the model answers safe matched prompts.

2. Over-Refusal Rate
   - Fraction of safe prompts incorrectly refused.

3. Output Length
   - Needed to ensure pruning is not merely shortening responses.

4. Refusal Template Frequency
   - Needed to detect whether the model is just becoming more generic or hesitant.

### Central Metric

Use:

```text
Safety Selectivity Index = drop in harmful compliance - drop in benign helpfulness
```

The paper should center this metric.

Also plot a Pareto frontier:

```text
x-axis: benign helpfulness drop
y-axis: harmful compliance reduction
```

The proposed method should dominate random pruning, magnitude pruning, and always-on pruning.

## Baselines

Compare against:

1. Dense model, no pruning.
2. Random structured FFN pruning.
3. Magnitude-based FFN pruning.
4. Activation-based static pruning.
5. Always-on pruning using the same mask.
6. Pattern-triggered pruning, proposed method.
7. Optional: external prompt classifier plus refusal baseline.

The key control is random pruning. Without it, reviewers can say the model simply got worse.

## Model Choices

Start with one model for a fast pilot:

- mistralai/Mistral-7B-Instruct-v0.3.

This is a good pilot choice because it is already listed in `configs/defuse_experiment_config.yaml` and the repo has Mistral hook support in `src/hook_setup.py`.

Before running the pilot, update the model config carefully:

```yaml
model:
  name: "mistralai/Mistral-7B-Instruct-v0.3"
  cache_dir: "<local path for Mistral weights>"
```

The current config may point to a Qwen cache path, so do not reuse that path blindly.

Important caveat:

If the dense Mistral model refuses nearly all harmful prompts, the pilot will not be informative. In that case, use harder jailbreak prompts or switch to a model with enough baseline harmful compliance to measure reduction.

For the paper, use at least two model families:

- Mistral family
- Llama family
- Qwen family
- Optional smaller control model

The goal is to show the effect is not model-specific.

## Experiments

### Experiment 1: Pattern Separability

Question:

Can harmful and benign prompts be separated using internal activations?

Procedure:

- Collect FFN activations.
- Train simple detectors per layer.
- Report AUROC, accuracy, precision, recall.
- Identify layers with strongest harmful-vs-benign separation.

Expected strong result:

Harmful prompts are separable from benign matched prompts using mid/late FFN activations.

### Experiment 2: Fragility Curves

Question:

Does harmful compliance degrade faster than benign helpfulness under pruning?

Procedure:

- Sweep sparsity: 0%, 30%, 50%, 70%.
- Compare harmful compliance and benign helpfulness.
- Plot degradation curves.

Expected strong result:

Harmful compliance drops sharply while benign helpfulness drops slowly.

### Experiment 3: Pattern-Triggered Pruning

Question:

Does conditional pruning outperform always-on and random pruning?

Procedure:

- Detect harmful activation pattern during inference.
- Trigger pruning only when pattern appears.
- Compare against baselines.

Expected strong result:

Pattern-triggered pruning reduces harmful compliance and jailbreak ASR while preserving benign helpfulness better than always-on pruning.

### Experiment 4: Jailbreak Robustness

Question:

Does the method work when harmful prompts are wrapped in jailbreak templates?

Procedure:

- Evaluate on JailbreakBench and StrongREJECT.
- Compare direct harmful vs jailbreak-wrapped harmful prompts.

Expected strong result:

Jailbreak ASR drops without a large over-refusal spike.

### Experiment 5: Causal Verification

Question:

Are the detected activations actually causing harmful compliance?

Procedure:

- Ablate candidate neurons/layers.
- Use activation patching or attribution patching on selected examples.
- Test whether disrupting the detected pattern changes the output.

Expected strong result:

Intervening on the detected harmful pattern changes harmful compliance behavior more than intervening on random activations.

## Minimal Pilot

Before full-scale experiments, run a fast pilot:

1. One model: `mistralai/Mistral-7B-Instruct-v0.3`.
2. HarmBench harmful direct: 100 prompts.
3. JailbreakBench benign: 100 prompts.
4. XSTest benign-sensitive: 100 prompts.
5. Optional after baseline works: JailbreakBench harmful wrapped, 100 prompts.
6. Capture FFN activations from selected layers.
7. Train harmful-vs-benign linear probe.
8. Implement top-k harmful-neuron pruning.
9. Compare:
   - dense
   - random pruning
   - always-on harmful-neuron pruning
   - pattern-triggered pruning

If the pilot does not show activation separability or safety selectivity, do not scale yet.

## Practical Execution Phases

### Phase 1: Dataset and Baseline Setup

Goal:

Create the pilot dataset and make sure the dense model runs cleanly on it.

Tasks:

1. Confirm `mistralai/Mistral-7B-Instruct-v0.3` loads correctly.
2. Update `configs/defuse_experiment_config.yaml` with the correct Mistral model path.
3. Download or load:
   - HarmBench
   - JailbreakBench benign
   - XSTest
4. Select 100 prompts from each dataset using the stratified rules above.
5. Save selected prompt IDs and prompt text in a reproducible local file.
6. Create train/validation/test splits:
   - 60 train
   - 20 validation
   - 20 test
7. Make sure every prompt has metadata:
   - `prompt_id`
   - `dataset`
   - `split`
   - `category`
   - `safety_label`
   - `prompt`

Recommended output file:

```text
datasets/pilot_safety_selectivity/pilot_prompts.jsonl
```

Each row should look like:

```json
{"prompt_id":"harmbench_0001","dataset":"harmbench","split":"train","category":"cyber","safety_label":"harmful","prompt":"..."}
```

Phase 1 is complete when the prompt file exists, is reproducible, and the Mistral model can be loaded with the local config.

### Phase 2: Dense Baseline Generation

Goal:

Run the unpruned dense model on the pilot prompts and measure baseline behavior.

Generation settings:

```text
temperature = 0
top_p = 1
max_new_tokens = 256 or 512
same system prompt for every condition
same chat template for every condition
```

For each prompt, save:

```text
prompt_id
dataset
split
category
safety_label
prompt
model_name
decoding_settings
model_response
output_length
refusal_heuristic
judge_score if available
```

Recommended output file:

```text
results/pilot_safety_selectivity/dense_baseline_generations.jsonl
```

Minimum metrics to report:

1. HarmBench harmful compliance rate.
2. HarmBench refusal rate.
3. JailbreakBench benign refusal rate.
4. XSTest over-refusal rate.
5. Average output length by dataset.
6. A few representative examples of:
   - harmful compliance
   - correct refusal
   - benign helpful answer
   - over-refusal

Phase 2 is complete when dense baseline generations and the initial metric summary are saved.

Do not start activation capture, linear probes, pruning, or pattern-triggered pruning until Phase 2 has been reviewed.

### Phase 3: Activation Collection

Goal:

Capture FFN activations for the pilot prompts without pruning.

Initial activation targets:

1. Input to MLP down projection, usually `pre_mlp2` / `down_proj` input.
2. Last prompt token activation.
3. First generated token activation.
4. Mean over prompt tokens as a fallback representation.

Phase 3 should only begin after Phase 2 results show that the dense model has measurable harmful compliance and reasonable benign helpfulness.

### Phase 4: Pattern Separability

Goal:

Check whether harmful and benign prompts are separable using internal activations.

Start with:

1. Centroid similarity detector.
2. Logistic regression linear probe.

Report per layer:

```text
AUROC
accuracy
precision
recall
false positive rate on XSTest
```

### Phase 5: Static Pruning Sanity Check

Goal:

Test whether harmful-associated FFN neuron pruning beats random pruning.

Neuron score:

```text
score_neuron = mean_abs_activation_harmful - mean_abs_activation_benign
```

Sparsity sweep:

```text
10%, 20%, 30%, 50%
```

Compare:

1. dense
2. random pruning
3. magnitude pruning
4. harmful-score pruning

### Phase 6: Pattern-Triggered Pruning

Goal:

Make pruning conditional on harmful activation pattern detection.

Trigger:

```text
if harmful_detector_score > threshold:
    apply harmful-neuron pruning mask
else:
    no pruning
```

Thresholds must be chosen on validation data only.

### Phase 7: Causal Evidence

Goal:

Show that detected harmful activations are causally involved in harmful compliance.

Run:

1. Candidate neuron ablation.
2. Random neuron ablation.
3. Benign-associated neuron ablation.
4. Small activation patching study on selected examples.

## Main Risks

### Risk 1: Detector learns dataset artifacts

Mitigation:

- Train on one dataset, test on another.
- Train on direct harmful, test on jailbreak.
- Use matched benign controls.

### Risk 2: Pruning just damages the model

Mitigation:

- Include benign helpfulness and over-refusal metrics.
- Compare against random pruning.
- Track output length and refusal-template frequency.

### Risk 3: Harmful and benign activations overlap too much

Mitigation:

- Use layer-wise detectors.
- Try co-activation patterns instead of individual neurons.
- Try probe-based detection before pruning.

### Risk 4: Method over-refuses

Mitigation:

- Evaluate on XSTest and OR-Bench.
- Require multi-layer trigger agreement.
- Tune threshold on validation set, not test set.

## Paper Structure

### Title Options

1. Detecting and Disrupting Harmful Computation via Pattern-Triggered Pruning
2. Conditional Activation Signatures and the Fragility of Harmful Behavior in Large Language Models
3. Pattern-Triggered FFN Pruning for Safety-Selective Inference in LLMs

### Contributions

1. We introduce pattern-triggered FFN pruning, an inference-time method that conditionally suppresses harmful activation patterns.
2. We show harmful compliance has detectable activation signatures in aligned LLMs.
3. We show harmful compliance is more fragile than benign helpfulness under structured FFN perturbation.
4. We evaluate safety selectivity across direct harmful prompts, jailbreak prompts, benign matched prompts, and over-refusal benchmarks.
5. We provide causal evidence that the detected patterns contribute to harmful compliance.

## Success Criteria

The project is strong if:

- harmful compliance drops materially
- jailbreak ASR drops materially
- benign helpfulness drops only modestly
- over-refusal does not spike
- random pruning does not reproduce the result
- activation detectors generalize across datasets
- results hold across at least two model families

The project is weak if:

- harmful and benign performance collapse together
- the detector only works on seen datasets
- gains come from shorter or generic refusals
- always-on pruning performs as well as pattern-triggered pruning

## Immediate Next Steps

1. Complete Phase 1: dataset and baseline setup.
2. Complete Phase 2: dense baseline generation.
3. Review dense baseline results before moving to activation capture.
4. Only if Phase 2 is promising, start Phase 3 activation collection.

## Team Handoff

People involved:

- Abhishek
- Naren
- Tapasvi

### Tapasvi: Immediate Assignment

Tapasvi should own only Phase 1 and Phase 2 for now.

Scope:

1. Build the pilot prompt file.
2. Configure and verify `mistralai/Mistral-7B-Instruct-v0.3`.
3. Run dense baseline generation.
4. Save the dense model outputs.
5. Produce the initial metric summary.

Tapasvi should not go beyond Phase 2.

Specifically, Tapasvi should not start:

- activation capture
- linear probes
- harmful activation signature extraction
- static pruning
- pattern-triggered pruning
- causal intervention experiments

Once Phase 2 is complete, Tapasvi should report back with:

1. Location of the selected prompt file.
2. Location of the dense baseline output file.
3. Confirmation that Mistral loaded from the expected cache path.
4. Dense baseline metric summary:
   - HarmBench harmful compliance rate
   - HarmBench refusal rate
   - JailbreakBench benign refusal rate
   - XSTest over-refusal rate
   - average output length per dataset
5. 5 to 10 representative model outputs.
6. Any loading, dataset, formatting, or judgment issues encountered.

After that, Naren and Abhishek should review whether the dense baseline is informative enough to proceed to Phase 3.
