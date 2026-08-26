"""
Phase 4: Are harmful and benign prompts separable from FFN activations?

Approved by Naren via WhatsApp, 2026-08-06, after reviewing Phase 3 results
(see progress.md).

Implements Experiment 1 / Phase 4 of the plan: for every layer and every
captured activation point, fit two detectors on the train split and report
per-layer separability.

  1. Centroid detector — harmful_direction = mean(harmful) - mean(benign),
     score = cosine(activation, harmful_direction). No fitted parameters
     beyond the two class means.
  2. Logistic-regression probe — standardized activations, L2-regularized.

Split discipline (plan: "Do not tune thresholds ... on the test split"):
  train (180 prompts) — fit centroids / probe weights
  val   (60)          — pick the decision threshold
  test  (60)          — reported once, using the val-chosen threshold

Reported per layer × activation point: AUROC, accuracy, precision, recall,
and false-positive rate on XSTest specifically (the over-refusal-risk set,
where a false positive means a safe prompt would wrongly trigger pruning).

Inputs:  results/pilot_safety_selectivity/activations/pilot_activations_aggregated.pt
Outputs: results/pilot_safety_selectivity/separability_metrics.json
         results/pilot_safety_selectivity/separability_summary.md
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AGG_FILE = Path("results/pilot_safety_selectivity/activations/pilot_activations_aggregated.pt")
OUT_DIR = Path("results/pilot_safety_selectivity")
METRICS_FILE = OUT_DIR / "separability_metrics.json"
SUMMARY_FILE = OUT_DIR / "separability_summary.md"

ACTIVATION_POINTS = ["last_prompt_token", "mean_prompt_tokens", "first_generated_token"]
SEED = 42


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def classification_metrics(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return {
        "accuracy": (tp + tn) / max(len(y_true), 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def pick_threshold(scores, y_true):
    """Threshold maximizing Youden's J (recall - FPR) on the validation split."""
    candidates = np.unique(scores)
    best_thr, best_j = float(candidates[0]), -np.inf
    for thr in candidates:
        pred = (scores >= thr).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        tn = ((pred == 0) & (y_true == 0)).sum()
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        j = tpr - fpr
        if j > best_j:
            best_j, best_thr = j, float(thr)
    return best_thr


def evaluate(scores, y_true, thr, is_xstest):
    pred = (scores >= thr).astype(int)
    m = classification_metrics(y_true, pred)
    m["auroc"] = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan")
    xs = is_xstest & (y_true == 0)
    m["xstest_fpr"] = float(pred[xs].mean()) if xs.sum() else float("nan")
    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-only", action="store_true",
                        help="Rebuild separability_summary.md from the existing "
                             "separability_metrics.json without refitting anything.")
    args = parser.parse_args()

    if args.summary_only:
        with open(METRICS_FILE) as f:
            results = json.load(f)
        write_summary(results)
        return

    print(f"Loading {AGG_FILE}")
    agg = torch.load(AGG_FILE)

    split = np.array(agg["split"])
    y = (np.array(agg["safety_label"]) == "harmful").astype(int)
    is_xstest = np.array(agg["dataset"]) == "xstest"

    tr, va, te = split == "train", split == "val", split == "test"
    print(f"train={tr.sum()} val={va.sum()} test={te.sum()} | harmful={y.sum()} benign={(1 - y).sum()}")

    num_layers = agg[ACTIVATION_POINTS[0]].shape[1]
    results = {}

    for point in ACTIVATION_POINTS:
        acts = agg[point].float().numpy()  # [N, num_layers, intermediate_size]
        per_layer = []

        for layer in tqdm(range(num_layers), desc=point):
            X = acts[:, layer, :]
            X_tr, y_tr = X[tr], y[tr]

            # --- centroid detector ---
            direction = X_tr[y_tr == 1].mean(axis=0) - X_tr[y_tr == 0].mean(axis=0)
            norm = np.linalg.norm(X, axis=1) * np.linalg.norm(direction) + 1e-8
            cos = (X @ direction) / norm
            c_thr = pick_threshold(cos[va], y[va])
            centroid = {
                "val": evaluate(cos[va], y[va], c_thr, is_xstest[va]),
                "test": evaluate(cos[te], y[te], c_thr, is_xstest[te]),
                "threshold": c_thr,
            }

            # --- logistic-regression probe ---
            scaler = StandardScaler().fit(X_tr)
            clf = LogisticRegression(max_iter=2000, random_state=SEED)
            clf.fit(scaler.transform(X_tr), y_tr)
            prob = clf.predict_proba(scaler.transform(X))[:, 1]
            p_thr = pick_threshold(prob[va], y[va])
            probe = {
                "val": evaluate(prob[va], y[va], p_thr, is_xstest[va]),
                "test": evaluate(prob[te], y[te], p_thr, is_xstest[te]),
                "threshold": p_thr,
            }

            per_layer.append({"layer": layer, "centroid": centroid, "probe": probe})

        results[point] = per_layer

    results["cross_source"] = cross_source_check(
        agg, y, is_xstest, train_datasets=("harmbench", "jailbreakbench_benign"),
        heldout_dataset="xstest", heldout_mask=is_xstest,
    )

    # Item 5 of Naren's 2026-08-24 checklist: the reverse direction of the
    # source-shift control — train on HarmBench + XSTest, hold out
    # JailbreakBench-benign this time, instead of XSTest.
    is_jbb = np.array(agg["dataset"]) == "jailbreakbench_benign"
    results["cross_source_reverse"] = cross_source_check(
        agg, y, is_xstest, train_datasets=("harmbench", "xstest"),
        heldout_dataset="jailbreakbench_benign", heldout_mask=is_jbb,
    )

    # Item 4 of the checklist: output-grounded separability. In-distribution
    # separability above uses the *source label* (prompt came from HarmBench)
    # as the positive class. That conflates "is a HarmBench prompt" with "the
    # model actually gave harmful help" — a HarmBench prompt the model
    # refused is not a harmful compliance. Redefine positives using the
    # HarmBench classifier's actual verdict on the generation and re-run the
    # same detectors, reported separately.
    results["output_grounded"] = output_grounded_check(agg, split, is_xstest)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics: {METRICS_FILE}")

    write_summary(results)


def cross_source_check(agg, y, is_xstest, train_datasets, heldout_dataset, heldout_mask):
    """Risk 1 mitigation (plan: "Train on one dataset, test on another").

    In-distribution separability is confounded: harmful comes only from
    HarmBench, so a probe can score well by learning *which corpus* a prompt
    came from rather than whether it is harmful. Train on two of the three
    corpora, then measure the false-positive rate on the held-out third —
    a benign corpus the probe has never seen. `is_xstest` is passed through
    for backward-compatible field naming even when XSTest is the training
    corpus, not the held-out one (see cross_source_reverse below).
    """
    dataset = np.array(agg["dataset"])
    train_mask = np.isin(dataset, train_datasets)
    out = {}

    for point in ACTIVATION_POINTS:
        acts = agg[point].float().numpy()
        per_layer = []
        for layer in tqdm(range(acts.shape[1]), desc=f"cross-source[{heldout_dataset}] {point}"):
            X = acts[:, layer, :]
            scaler = StandardScaler().fit(X[train_mask])
            clf = LogisticRegression(max_iter=2000, random_state=SEED)
            clf.fit(scaler.transform(X[train_mask]), y[train_mask])
            prob = clf.predict_proba(scaler.transform(X))[:, 1]
            per_layer.append({
                "layer": layer,
                "heldout_dataset": heldout_dataset,
                "xstest_fpr_unseen_source": float((prob[heldout_mask] >= 0.5).mean()),
                "train_auroc": float(roc_auc_score(y[train_mask], prob[train_mask])),
            })
        out[point] = per_layer
    return out


def output_grounded_check(agg, split, is_xstest):
    """Item 4 of the 2026-08-24 checklist.

    Redefines the positive class as "the HarmBench classifier says this
    generation actually carried out the harmful behavior", not "this prompt
    came from HarmBench". Every HarmBench prompt the dense model refused
    (per the classifier, not the phrase heuristic) becomes a negative. Benign
    corpora are unaffected — they were never candidates for the positive
    class. Uses the same train/val/test splits and detectors as the main run.
    """
    scores_file = OUT_DIR / "harmbench_classifier_scores.jsonl"
    if not scores_file.exists():
        print(f"WARNING: {scores_file} not found, skipping output-grounded check "
              "(run src/harmbench_rescore.py first).")
        return None

    label_by_id = {}
    with open(scores_file) as f:
        for line in f:
            r = json.loads(line)
            if r["harmbench_classifier_label"] is not None:
                label_by_id[r["prompt_id"]] = r["harmbench_classifier_label"] == "yes"

    prompt_ids = list(agg["prompt_id"])
    dataset = np.array(agg["dataset"])
    y_out = np.zeros(len(prompt_ids), dtype=int)
    covered = np.zeros(len(prompt_ids), dtype=bool)
    for i, (pid, ds) in enumerate(zip(prompt_ids, dataset)):
        if ds == "harmbench":
            if pid in label_by_id:
                y_out[i] = int(label_by_id[pid])
                covered[i] = True
            # else: no classifier verdict for this prompt_id, excluded below
        else:
            covered[i] = True  # benign corpora: negative by construction

    n_dropped = int((~covered).sum())
    if n_dropped:
        print(f"output_grounded_check: dropping {n_dropped} harmbench rows "
              "with no classifier verdict")

    tr = (split == "train") & covered
    va = (split == "val") & covered
    te = (split == "test") & covered
    print(f"output_grounded: train={tr.sum()} val={va.sum()} test={te.sum()} "
          f"| positives(actual compliance)={y_out[covered].sum()} "
          f"negatives={ (1 - y_out[covered]).sum() }")

    num_layers = agg[ACTIVATION_POINTS[0]].shape[1]
    out = {}
    for point in ACTIVATION_POINTS:
        acts = agg[point].float().numpy()
        per_layer = []
        for layer in tqdm(range(num_layers), desc=f"output-grounded {point}"):
            X = acts[:, layer, :]
            X_tr, y_tr = X[tr], y_out[tr]
            if len(np.unique(y_tr)) < 2:
                per_layer.append({"layer": layer, "skipped": "single class in train split"})
                continue

            scaler = StandardScaler().fit(X_tr)
            clf = LogisticRegression(max_iter=2000, random_state=SEED)
            clf.fit(scaler.transform(X_tr), y_tr)
            prob = clf.predict_proba(scaler.transform(X))[:, 1]
            p_thr = pick_threshold(prob[va], y_out[va])
            per_layer.append({
                "layer": layer,
                "probe": {
                    "val": evaluate(prob[va], y_out[va], p_thr, is_xstest[va]),
                    "test": evaluate(prob[te], y_out[te], p_thr, is_xstest[te]),
                    "threshold": p_thr,
                },
            })
        out[point] = per_layer
    return out


def write_summary(results):
    lines = ["# Phase 4: Pattern Separability — Summary", ""]
    lines.append("Test-split metrics at the validation-chosen threshold. "
                 "`xstest_fpr` = fraction of safe XSTest prompts flagged harmful.")
    lines.append("")

    EXTRA_KEYS = ("cross_source", "cross_source_reverse", "output_grounded")

    best_overall = None
    for point, per_layer in results.items():
        if point in EXTRA_KEYS:
            continue
        lines.append(f"## {point}")
        lines.append("")
        lines.append("| layer | centroid AUROC | probe AUROC | probe acc | probe prec | probe rec | probe XSTest FPR |")
        lines.append("|-------|----------------|-------------|-----------|------------|-----------|------------------|")
        for row in per_layer:
            c, p = row["centroid"]["test"], row["probe"]["test"]
            lines.append(
                f"| {row['layer']} | {c['auroc']:.3f} | {p['auroc']:.3f} | {p['accuracy']:.3f} | "
                f"{p['precision']:.3f} | {p['recall']:.3f} | {p['xstest_fpr']:.3f} |"
            )
            cand = (p["auroc"], point, row["layer"])
            if best_overall is None or cand[0] > best_overall[0]:
                best_overall = cand
        lines.append("")

    auroc, point, layer = best_overall
    lines.append(f"**Best probe AUROC (test): {auroc:.3f} — {point}, layer {layer}.**")
    lines.append("")

    if "cross_source" in results:
        lines.append("## Confound check (plan Risk 1: dataset artifacts)")
        lines.append("")
        lines.append("Harmful prompts come only from HarmBench, so in-distribution "
                     "separability may reflect *corpus identity*, not harmfulness. "
                     "Below: probe trained with JailbreakBench as the only benign "
                     "source, then applied to XSTest — a benign corpus it never saw. "
                     "A high FPR here means the probe flags safe prompts once the "
                     "benign source changes.")
        lines.append("")
        lines.append("| activation point | layer 0 probe AUROC (in-dist test) | XSTest FPR, unseen source (layer 0) | best-layer XSTest FPR, unseen source |")
        lines.append("|------------------|-----------------------------------|-------------------------------------|--------------------------------------|")
        for pt in ACTIVATION_POINTS:
            l0_auroc = results[pt][0]["probe"]["test"]["auroc"]
            cs = results["cross_source"][pt]
            l0_fpr = cs[0]["xstest_fpr_unseen_source"]
            best_fpr = min(r["xstest_fpr_unseen_source"] for r in cs)
            lines.append(f"| {pt} | {l0_auroc:.3f} | {l0_fpr:.3f} | {best_fpr:.3f} |")
        lines.append("")
        lines.append("Layer 0's down_proj input is essentially a function of token "
                     "embeddings. Whatever layer 0 achieves is obtainable from surface "
                     "form alone, so treat it as the floor that deeper layers must beat "
                     "for the 'harmful computation' claim to hold.")
        lines.append("")
        lines.append("### Layer choice does not transfer")
        lines.append("")
        lines.append("| activation point | best in-dist layer | its AUROC | its unseen-source FPR | lowest-FPR layer | that layer's FPR |")
        lines.append("|------------------|--------------------|-----------|-----------------------|------------------|------------------|")
        for pt in ACTIVATION_POINTS:
            per_layer, cs = results[pt], results["cross_source"][pt]
            bi = max(per_layer, key=lambda r: r["probe"]["test"]["auroc"])
            bl = min(cs, key=lambda r: r["xstest_fpr_unseen_source"])
            lines.append(
                f"| {pt} | {bi['layer']} | {bi['probe']['test']['auroc']:.3f} | "
                f"{cs[bi['layer']]['xstest_fpr_unseen_source']:.3f} | {bl['layer']} | "
                f"{bl['xstest_fpr_unseen_source']:.3f} |"
            )
        lines.append("")
        lines.append("The layer that separates best in-distribution is **not** the layer "
                     "that holds up when the benign source changes. Selecting a trigger "
                     "layer by in-distribution AUROC would pick one of the worst layers "
                     "for over-refusal risk — relevant to the Phase 5-6 trigger design.")
        lines.append("")

    if "cross_source_reverse" in results:
        lines.append("## Reverse source-shift control (checklist item 5, 2026-08-24)")
        lines.append("")
        lines.append("Mirror of the check above: train on HarmBench + XSTest, hold out "
                     "JailbreakBench-benign this time (the earlier check held out XSTest). "
                     "Complements the first direction rather than replacing it.")
        lines.append("")
        lines.append("| activation point | layer 0 FPR, held-out JBB-benign | best-layer FPR, held-out JBB-benign |")
        lines.append("|------------------|-----------------------------------|--------------------------------------|")
        for pt in ACTIVATION_POINTS:
            cs = results["cross_source_reverse"][pt]
            l0_fpr = cs[0]["xstest_fpr_unseen_source"]
            best_fpr = min(r["xstest_fpr_unseen_source"] for r in cs)
            lines.append(f"| {pt} | {l0_fpr:.3f} | {best_fpr:.3f} |")
        lines.append("")

    if "output_grounded" in results and results["output_grounded"] is not None:
        lines.append("## Output-grounded separability (checklist item 4, 2026-08-24)")
        lines.append("")
        lines.append("Everything above uses *prompt source* as the positive class — a "
                     "HarmBench prompt is 'harmful' whether or not the model actually "
                     "complied. This redefines positives as 'the HarmBench classifier "
                     "verdict says this generation actually carried out the harmful "
                     "behavior' (45/100 per the re-score), so a refused HarmBench prompt "
                     "is now correctly a negative. Same detectors, splits, procedure.")
        lines.append("")
        lines.append("| activation point | best layer | probe AUROC (test) | probe acc | probe XSTest FPR |")
        lines.append("|-------------------|-----------|---------------------|-----------|-------------------|")
        og_best = None
        for pt in ACTIVATION_POINTS:
            scored = [r for r in results["output_grounded"][pt] if "probe" in r]
            if not scored:
                lines.append(f"| {pt} | - | skipped (no valid layers) | - | - |")
                continue
            best = max(scored, key=lambda r: r["probe"]["test"]["auroc"])
            t = best["probe"]["test"]
            lines.append(f"| {pt} | {best['layer']} | {t['auroc']:.3f} | {t['accuracy']:.3f} | {t['xstest_fpr']:.3f} |")
            cand = (t["auroc"], pt, best["layer"])
            if og_best is None or cand[0] > og_best[0]:
                og_best = cand
        lines.append("")
        if og_best:
            lines.append(f"Best output-grounded probe AUROC (test): {og_best[0]:.3f} "
                         f"({og_best[1]}, layer {og_best[2]}). Compare against the "
                         f"source-label best of {auroc:.3f} above — a materially lower "
                         "number here would mean prior sections were separating corpus "
                         "identity, not harmful compliance.")
        lines.append("")

    with open(SUMMARY_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved summary: {SUMMARY_FILE}")
    print(f"\nBest probe AUROC (test): {auroc:.3f} — {point}, layer {layer}")


if __name__ == "__main__":
    main()
