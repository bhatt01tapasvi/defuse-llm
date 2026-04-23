"""
Phase 1: Build pilot prompt dataset for safety selectivity experiments.

Outputs: datasets/pilot_safety_selectivity/pilot_prompts.jsonl
         datasets/pilot_safety_selectivity/dataset_stats.json

Sources:
  - HarmBench (direct/standard behaviors) - 100 harmful prompts
  - JailbreakBench benign behaviors - 100 safe prompts
  - XSTest safe-sensitive prompts - 100 safe prompts

Each row schema:
  prompt_id, dataset, split, category, safety_label, prompt
"""

import csv
import importlib.util
import io
import json
import os
import random
import ssl
import urllib.request
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

SEED = 42
OUT_DIR = Path("datasets/pilot_safety_selectivity")
OUT_FILE = OUT_DIR / "pilot_prompts.jsonl"
STATS_FILE = OUT_DIR / "dataset_stats.json"

SPLITS = {"train": 60, "val": 20, "test": 20}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assign_splits(rows, seed=SEED):
    """Assign train/val/test splits stratified by category."""
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    result = []
    rng = random.Random(seed)
    for cat, items in by_cat.items():
        rng.shuffle(items)
        n = len(items)
        n_train = round(n * 0.6)
        n_val = round(n * 0.2)
        for i, item in enumerate(items):
            if i < n_train:
                item["split"] = "train"
            elif i < n_train + n_val:
                item["split"] = "val"
            else:
                item["split"] = "test"
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# HarmBench
# ---------------------------------------------------------------------------

HARMBENCH_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
    "main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
)

# Map HarmBench SemanticCategory to pilot target categories
HARMBENCH_CAT_MAP = {
    "cybercrime_intrusion": "cyber",
    "chemical_biological": "chemical_bio",
    "misinformation_disinformation": "misinfo",
    "harassment_bullying": "hate",
    "harmful": "violence",
    "illegal": None,  # split into multiple target categories below
}

ILLEGAL_SUBCATS = ["fraud", "weapons", "privacy", "self_harm", "illegal_activity"]


def load_harmbench(n=100, seed=SEED):
    print("Loading HarmBench from GitHub...")
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(HARMBENCH_URL, context=ctx, timeout=30) as r:
        data = r.read().decode()

    reader = csv.DictReader(io.StringIO(data))
    rows = [r for r in reader if r["FunctionalCategory"] == "standard"]
    print(f"  Found {len(rows)} standard behaviors")

    rng = random.Random(seed)

    # Group by semantic category
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["SemanticCategory"]].append(r)

    selected = []
    counters = {}

    # Fixed categories: take 10 each
    for sem_cat, tgt_cat in HARMBENCH_CAT_MAP.items():
        if tgt_cat is None:
            continue
        pool = by_cat[sem_cat]
        rng.shuffle(pool)
        take = min(10, len(pool))
        for item in pool[:take]:
            selected.append({
                "category": tgt_cat,
                "prompt": item["Behavior"],
                "source_id": item["BehaviorID"],
            })
        counters[tgt_cat] = take

    # "illegal" category: split into 5 subcategories × 10
    illegal_pool = list(by_cat["illegal"])
    rng.shuffle(illegal_pool)
    per_subcat = len(ILLEGAL_SUBCATS)
    chunk = len(illegal_pool) // per_subcat
    for i, subcat in enumerate(ILLEGAL_SUBCATS):
        start = i * chunk
        items = illegal_pool[start: start + 10]
        for item in items:
            selected.append({
                "category": subcat,
                "prompt": item["Behavior"],
                "source_id": item["BehaviorID"],
            })
        counters[subcat] = len(items)

    print(f"  Selected {len(selected)} HarmBench prompts")
    print(f"  Category counts: {counters}")

    result = []
    for idx, item in enumerate(selected):
        result.append({
            "prompt_id": f"harmbench_{idx+1:04d}",
            "dataset": "harmbench",
            "split": "",
            "category": item["category"],
            "safety_label": "harmful",
            "prompt": item["prompt"],
        })
    return result


# ---------------------------------------------------------------------------
# JailbreakBench benign
# ---------------------------------------------------------------------------

JBB_PKG_PATH = None  # resolved at runtime


def _get_jbb_dataset_path():
    import site
    for sp in site.getsitepackages():
        p = Path(sp) / "jailbreakbench" / "dataset.py"
        if p.exists():
            return str(p)
    return None


def load_jailbreakbench_benign(n=100, seed=SEED):
    print("Loading JailbreakBench benign...")
    path = _get_jbb_dataset_path()
    if path is None:
        raise RuntimeError("jailbreakbench dataset.py not found in site-packages")

    spec = importlib.util.spec_from_file_location("jbb_dataset", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ds = mod.read_dataset("benign")
    df = ds.as_dataframe()
    print(f"  Found {len(df)} benign behaviors")
    print(f"  Categories: {df['Category'].value_counts().to_dict()}")

    # Map JBB categories to pilot target categories
    jbb_to_pilot = {
        "Harassment/Discrimination": "benign_writing_persuasion",
        "Malware/Hacking": "benign_cybersec",
        "Physical harm": "benign_health",
        "Economic harm": "benign_finance",
        "Fraud/Deception": "benign_legal_policy",
        "Disinformation": "benign_history_politics",
        "Sexual/Adult content": "benign_writing_persuasion",
        "Privacy": "benign_privacy",
        "Expert advice": "benign_general_tech",
        "Government decision-making": "benign_legal_policy",
    }

    rng = random.Random(seed)
    result = []
    idx = 0
    for _, row in df.iterrows():
        tgt = jbb_to_pilot.get(row["Category"], "benign_general_tech")
        result.append({
            "prompt_id": f"jbb_benign_{idx+1:04d}",
            "dataset": "jailbreakbench_benign",
            "split": "",
            "category": tgt,
            "safety_label": "benign",
            "prompt": row["Goal"],
        })
        idx += 1

    print(f"  Selected {len(result)} JBB benign prompts")
    return result


# ---------------------------------------------------------------------------
# XSTest
# ---------------------------------------------------------------------------

XSTEST_SAFE_TYPES = [
    "homonyms",
    "figurative_language",
    "safe_targets",
    "safe_contexts",
    "definitions",
    "historical_events",
    "privacy_public",
    "privacy_fictional",
    "real_group_nons_discr",
    "nons_group_real_discr",
]

XSTEST_TYPE_TO_CAT = {
    "homonyms": "xs_homonyms",
    "figurative_language": "xs_figurative",
    "safe_targets": "xs_safe_targets",
    "safe_contexts": "xs_safe_contexts",
    "definitions": "xs_definitions",
    "historical_events": "xs_historical",
    "privacy_public": "xs_privacy",
    "privacy_fictional": "xs_privacy",
    "real_group_nons_discr": "xs_discrimination",
    "nons_group_real_discr": "xs_discrimination",
}


def load_xstest(n=100, seed=SEED):
    print("Loading XSTest...")
    ds = load_dataset("Paul/XSTest")["train"]
    df = ds.to_pandas()

    # Only safe prompts; exclude "contrast_" types (those test unsafe refusal)
    safe = df[df["label"] == "safe"].copy()
    print(f"  Found {len(safe)} safe prompts, types: {safe['type'].value_counts().to_dict()}")

    rng = random.Random(seed)

    # Stratified sample: 10 per type from XSTEST_SAFE_TYPES
    selected = []
    for xtype in XSTEST_SAFE_TYPES:
        pool = safe[safe["type"] == xtype]
        take = min(10, len(pool))
        sampled = pool.sample(n=take, random_state=seed)
        for _, row in sampled.iterrows():
            selected.append({
                "type": xtype,
                "prompt": row["prompt"],
                "source_id": str(row["id"]),
            })

    print(f"  Selected {len(selected)} XSTest prompts")

    result = []
    for idx, item in enumerate(selected):
        result.append({
            "prompt_id": f"xstest_{idx+1:04d}",
            "dataset": "xstest",
            "split": "",
            "category": XSTEST_TYPE_TO_CAT.get(item["type"], "xs_other"),
            "safety_label": "benign",
            "prompt": item["prompt"],
        })
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    harmbench = load_harmbench()
    jbb = load_jailbreakbench_benign()
    xstest = load_xstest()

    # Assign splits per dataset
    harmbench = assign_splits(harmbench)
    jbb = assign_splits(jbb)
    xstest = assign_splits(xstest)

    all_prompts = harmbench + jbb + xstest
    print(f"\nTotal prompts: {len(all_prompts)}")

    # Write JSONL
    with open(OUT_FILE, "w") as f:
        for row in all_prompts:
            f.write(json.dumps(row) + "\n")
    print(f"Saved: {OUT_FILE}")

    # Stats
    stats = {
        "total": len(all_prompts),
        "by_dataset": {},
        "by_split": {},
        "by_safety_label": {},
    }
    for row in all_prompts:
        stats["by_dataset"][row["dataset"]] = stats["by_dataset"].get(row["dataset"], 0) + 1
        stats["by_split"][row["split"]] = stats["by_split"].get(row["split"], 0) + 1
        stats["by_safety_label"][row["safety_label"]] = stats["by_safety_label"].get(row["safety_label"], 0) + 1

    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {STATS_FILE}")
    print(f"\nStats:\n{json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
