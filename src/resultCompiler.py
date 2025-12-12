from collections import defaultdict
import glob
import os 
import numpy as np
import pandas as pd
import json
import re
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LAYER_ANALYSIS_DIR = os.path.join(RESULTS_DIR, "layer_analysis")
MLP_IMPACT_DIR = os.path.join(RESULTS_DIR, "mlp_impact")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

def find_single_layer_dir(model_name: str, dataset_name: str, base_prune_ratio: float) -> Optional[str]:
    """Find directory matching pattern: {model_name}_{dataset_name}_single_*"""
    if base_prune_ratio > 0.0:
        pattern = os.path.join(LAYER_ANALYSIS_DIR, f"{model_name}_{dataset_name}_marginal_{base_prune_ratio}_*")
    else:
        pattern = os.path.join(LAYER_ANALYSIS_DIR, f"{model_name}_{dataset_name}_single_*")
    
    print(f"    Searching with pattern: {pattern}")
    matches = glob.glob(pattern)
    
    if not matches:
        print(f"    No matches with glob, trying manual search...")
        if os.path.exists(LAYER_ANALYSIS_DIR):
            all_dirs = [d for d in os.listdir(LAYER_ANALYSIS_DIR) 
                       if os.path.isdir(os.path.join(LAYER_ANALYSIS_DIR, d))]
            prefix = f"{model_name}_{dataset_name}_single_"
            matches = [os.path.join(LAYER_ANALYSIS_DIR, d) for d in all_dirs if d.startswith(prefix)]
            print(f"    Manual search found {len(matches)} matches")
    
    if not matches:
        return None
    
    return sorted(matches)[0]

def is_prune_dir(name: str) -> bool:
    """Check if directory name matches pruning pattern layer_<idx>_keep_<ratio>."""
    return re.match(r"^layer_\d+_keep_(0\.\d+|1\.0|0)$", name) is not None

def parse_prune_dir(name: str) -> Tuple[int, float]:
    """Extract layer index and keep ratio from directory name."""
    m = re.match(r"^layer_(\d+)_keep_(0\.\d+|1\.0|0)$", name)
    if not m:
        raise ValueError(f"Invalid prune dir: {name}")
    layer_idx = int(m.group(1))
    keep_ratio = float(m.group(2))
    return layer_idx, keep_ratio

def find_perplexity_in_dir(exp_dir: str, dataset_name: str) -> Optional[float]:
    """Search for perplexity score in JSON files within experiment directory."""
    patterns = ["perplexity_results.json"]
    
    for pattern in patterns:
        matches = glob.glob(os.path.join(exp_dir, "**", pattern), recursive=True)
        for file_path in matches:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                normalized_dataset = dataset_name.lower().replace("_", " ")
                
                for key, value in data.items():
                    normalized_key = key.lower().replace("_", " ")
                    
                    if (normalized_dataset == normalized_key or 
                        normalized_key.endswith(normalized_dataset)):
                        
                        if isinstance(value, dict):
                            if 'manual' in value and isinstance(value['manual'], (int, float)):
                                return float(value['manual'])
                            elif 'builtin' in value and isinstance(value['builtin'], (int, float)):
                                return float(value['builtin'])
                    
            except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                print(f"        Error reading {file_path}: {e}")
                continue
    
    return None

def find_mmlu_accuracy_in_dir(exp_dir: str, dataset_name: str) -> Optional[float]:
    """Search for MMLU accuracy in results_mmlu_*_shots.json files."""
    patterns = ["results_mmlu_*_shots.json"]
    
    for pattern in patterns:
        matches = glob.glob(os.path.join(exp_dir, "**", pattern), recursive=True)
        for file_path in matches:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Look in results dict for matching dataset
                if 'results' in data and isinstance(data['results'], dict):
                    normalized_dataset = dataset_name.lower().replace("_", " ")
                    
                    for key, value in data['results'].items():
                        normalized_key = key.lower().replace("_", " ")
                        
                        # Match mmlu_dataset_name with dataset_name
                        if (normalized_dataset == normalized_key or 
                            normalized_key.endswith(normalized_dataset)):
                            
                            if isinstance(value, dict):
                                # Extract accuracy from "acc,none" field
                                if 'acc,none' in value and isinstance(value['acc,none'], (int, float)):
                                    return float(value['acc,none'])
                                elif 'acc' in value and isinstance(value['acc'], (int, float)):
                                    return float(value['acc'])
                    
            except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                print(f"        Error reading {file_path}: {e}")
                continue
    
    return None

def collect_dataset_curves(base_dir: str, dataset_name: str) -> Tuple[Dict[float, List[Tuple[int, float]]], Dict[float, List[Tuple[int, float]]]]:
    """Collect both perplexity and MMLU accuracy curves for different keep ratios."""
    ppl_curves: Dict[float, List[Tuple[int, float]]] = {}
    mmlu_curves: Dict[float, List[Tuple[int, float]]] = {}
    
    if not os.path.isdir(base_dir):
        print(f"    ✗ Base directory does not exist: {base_dir}")
        return ppl_curves, mmlu_curves
    
    print(f"    Scanning directories in: {base_dir}")

    for d in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, d)
        if not os.path.isdir(dir_path) or not is_prune_dir(d):
            print(f"    Skipping non-prune dir: {d}")
            continue
        
        try:
            layer_idx, keep_ratio = parse_prune_dir(d)
            print(f"    Processing dir: {d} (layer {layer_idx}, keep {keep_ratio})")
            
            # Get perplexity
            ppl = find_perplexity_in_dir(dir_path, dataset_name)
            if ppl is not None:
                ppl_curves.setdefault(keep_ratio, []).append((layer_idx, ppl))
                print(f"      Perplexity: {ppl:.2f}")
            
            # Get MMLU accuracy
            mmlu_acc = find_mmlu_accuracy_in_dir(dir_path, dataset_name)
            if mmlu_acc is not None:
                mmlu_curves.setdefault(keep_ratio, []).append((layer_idx, mmlu_acc))
                print(f"      MMLU Accuracy: {mmlu_acc:.4f}")
                
        except Exception as e:
            print(f"      Warning: Error processing {d}: {e}")
            continue
    
    # Sort by layer index
    for keep_ratio in ppl_curves:
        ppl_curves[keep_ratio] = sorted(ppl_curves[keep_ratio], key=lambda x: x[0])
    for keep_ratio in mmlu_curves:
        mmlu_curves[keep_ratio] = sorted(mmlu_curves[keep_ratio], key=lambda x: x[0])
    
    return ppl_curves, mmlu_curves

def plot_curves(dataset_name: str, curves: Dict[float, List[Tuple[int, float]]], 
                model_name: str, baseline_val: Optional[float], 
                metric_name: str, ylabel: str):
    """Generic plotting function for both perplexity and MMLU accuracy."""
    if not curves:
        print(f"    No {metric_name} data to plot for {dataset_name}")
        return
    
    # Create directory structure
    dataset_plot_dir = os.path.join(PLOTS_DIR, "layer_analysis", dataset_name, "single")
    os.makedirs(dataset_plot_dir, exist_ok=True)
    
    palette = {
        1.0: "#1f77b4",
        0.75: "#2ca02c",
        0.5: "#ff7f0e",
        0.25: "#d62728",
        0.1: "#9467bd",
        0.0: "#8c564b",
    }
    
    plt.figure(figsize=(12, 7))
    
    all_layers = sorted(set(p[0] for points in curves.values() for p in points))
    
    # Plot baseline
    if baseline_val is not None:
        plt.axhline(y=baseline_val, color='black', linestyle='--', linewidth=2.5, 
                   label=f'Baseline (no pruning): {baseline_val:.4f}', alpha=0.7, zorder=1)
    
    for keep_ratio in sorted(curves.keys(), reverse=True):
        points = curves[keep_ratio]
        if not points:
            continue
        
        layers = [p[0] for p in points]
        vals = [p[1] for p in points]
        
        color = palette.get(keep_ratio, plt.cm.tab10(keep_ratio))
        label = f"keep {keep_ratio:.2f} ({int(keep_ratio*100)}%)"
        
        plt.plot(layers, vals, marker='o', linewidth=2.5, markersize=8, 
                label=label, color=color, alpha=0.85, zorder=2)
    
    plt.title(f"Layer-wise Pruning Impact on {metric_name}\n{dataset_name} ({model_name})", 
             fontsize=14, fontweight='bold')
    plt.xlabel("Pruned Layer Index", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(title="Neurons Kept", frameon=True, loc='best', fontsize=10)
    plt.tight_layout()
    
    all_ratios = sorted(curves.keys())
    layer_range = f"layer_{min(all_layers)}-{max(all_layers)}"
    ratio_str = "_".join([f"{r:.2f}".replace(".", "p") for r in all_ratios])
    filename = f"{metric_name.lower().replace(' ', '_')}_{layer_range}_keep_{ratio_str}.png"
    
    out_path = os.path.join(dataset_plot_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved {metric_name} plot: {out_path}")

def plot_single_layer_impact_self(model_name, dataset_name, base_prune_ratio):
    """Main plotting function for a single dataset - plots both perplexity and MMLU."""
    base_dir = find_single_layer_dir(model_name, dataset_name, base_prune_ratio)

    if not base_dir:
        print(f"  ✗ No directory found for {model_name}_{dataset_name}_single_*")
        return
    
    print(f"  ✓ Found: {os.path.basename(base_dir)}")
    
    # Collect both metrics
    ppl_curves, mmlu_curves = collect_dataset_curves(base_dir, dataset_name)
    
    # Get baselines
    baseline_dir = os.path.join(RESULTS_DIR, "baseline", f"{model_name}", "perplexity")
    baseline_ppl = find_perplexity_in_dir(baseline_dir, dataset_name)
    
    baseline_mmlu_dir = os.path.join(RESULTS_DIR, "baseline", f"{model_name}", "mmlu")
    baseline_mmlu = find_mmlu_accuracy_in_dir(baseline_mmlu_dir, dataset_name)
    
    if baseline_ppl is not None:
        print(f"    Baseline perplexity: {baseline_ppl:.2f}")
    else:
        print(f"    Warning: No baseline perplexity found")
    
    if baseline_mmlu is not None:
        print(f"    Baseline MMLU accuracy: {baseline_mmlu:.4f}")
    else:
        print(f"    Warning: No baseline MMLU accuracy found")
    
    # Plot perplexity
    if ppl_curves:
        total_ppl_points = sum(len(points) for points in ppl_curves.values())
        print(f"    Found {len(ppl_curves)} keep ratios, {total_ppl_points} perplexity points")
        plot_curves(dataset_name, ppl_curves, model_name, baseline_ppl, 
                   "Perplexity", "Perplexity")
    else:
        print(f"  ✗ No perplexity data found for {dataset_name}")
    
    # Plot MMLU accuracy
    if mmlu_curves:
        total_mmlu_points = sum(len(points) for points in mmlu_curves.values())
        print(f"    Found {len(mmlu_curves)} keep ratios, {total_mmlu_points} MMLU accuracy points")
        plot_curves(dataset_name, mmlu_curves, model_name, baseline_mmlu, 
                   "MMLU Accuracy", "Accuracy")
    else:
        print(f"  ✗ No MMLU accuracy data found for {dataset_name}")

def find_mlp_impact_csv(model_name: str, dataset_name: str) -> Optional[str]:
    """Find MLP impact CSV file for a given model and dataset."""
    model_dir = os.path.join(MLP_IMPACT_DIR, model_name)
    if not os.path.exists(model_dir):
        return None
    
    # Try different naming patterns
    patterns = [
        f"{dataset_name}mlp_impact.csv",
        f"{dataset_name}_mlp_impact.csv",
        f"custom_{dataset_name}_mlp_impact.csv",
        f"mmlu_{dataset_name}_mlp_impact.csv"
    ]
    
    for pattern in patterns:
        matches = glob.glob(os.path.join(model_dir, pattern))
        if matches:
            return matches[0]
    
    return None

def plot_cosine_similarity(model_name: str, dataset_name: str):
    """Plot cosine similarity scores across layers from MLP impact CSV."""
    csv_path = find_mlp_impact_csv(model_name, dataset_name)
    
    if not csv_path:
        print(f"  ✗ No MLP impact CSV found for {dataset_name}")
        return
    
    print(f"  ✓ Found CSV: {os.path.basename(csv_path)}")
    
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Extract layer numbers and cosine similarity metrics
        layers = []
        avg_cosine = []
        min_cosine = []
        max_cosine = []
        
        for _, row in df.iterrows():
            # Extract layer number from 'layer_X' format
            layer_str = row['layer']
            if isinstance(layer_str, str) and layer_str.startswith('layer_'):
                layer_num = int(layer_str.split('_')[1])
                layers.append(layer_num)
                avg_cosine.append(row['avg_cosine_sim'])
                min_cosine.append(row['min_cosine_sim'])
                max_cosine.append(row['max_cosine_sim'])
        
        if not layers:
            print(f"  ✗ No valid layer data found in CSV")
            return
        
        # Create plot directory
        dataset_plot_dir = os.path.join(PLOTS_DIR, "mlp_impact", dataset_name)
        os.makedirs(dataset_plot_dir, exist_ok=True)
        
        # Create single combined plot with all metrics
        plt.figure(figsize=(12, 7))
        plt.plot(layers, avg_cosine, marker='o', linewidth=2.5, markersize=8, 
                color='#1f77b4', label='Average', alpha=0.85, zorder=3)
        plt.plot(layers, min_cosine, marker='s', linewidth=2, markersize=6, 
                color='#d62728', label='Minimum', alpha=0.75, linestyle='--', zorder=2)
        plt.plot(layers, max_cosine, marker='^', linewidth=2, markersize=6, 
                color='#2ca02c', label='Maximum', alpha=0.75, linestyle='--', zorder=2)
        plt.fill_between(layers, min_cosine, max_cosine, alpha=0.15, color='gray', zorder=1)
        
        plt.title(f'MLP Output Cosine Similarity Across Layers\n{dataset_name} ({model_name})', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Layer Index', fontsize=12)
        plt.ylabel('Cosine Similarity', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.ylim([0, 1.05])
        plt.legend(frameon=True, loc='best', fontsize=10)
        plt.tight_layout()
        
        filename = f"cosine_similarity_{dataset_name}.png"
        out_path = os.path.join(dataset_plot_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Saved cosine similarity plot: {out_path}")
        
    except Exception as e:
        print(f"  ✗ Error plotting cosine similarity: {e}")

def plot_all_mlp_impacts(model_name: str):
    """Plot cosine similarity for all datasets with MLP impact data."""
    model_dir = os.path.join(MLP_IMPACT_DIR, model_name)
    
    if not os.path.exists(model_dir):
        print(f"✗ No MLP impact directory found for {model_name}")
        return
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(model_dir, "*_mlp_impact.csv"))
    
    if not csv_files:
        print(f"✗ No MLP impact CSV files found in {model_dir}")
        return
    
    print(f"\nFound {len(csv_files)} MLP impact CSV files for {model_name}\n")
    
    for csv_file in sorted(csv_files):
        # Extract dataset name from filename
        basename = os.path.basename(csv_file)
        # Remove prefixes and suffix
        dataset_name = basename.replace("_mlp_impact.csv", "")
        dataset_name = dataset_name.replace("custom_", "").replace("mmlu_", "")
        
        print(f"Processing: {dataset_name}")
        plot_cosine_similarity(model_name, dataset_name)
        print()

def main():
    models = ["meta-llama_Llama-3.2-3B"]
    datasets = [
        "college_computer_science", 
        "abstract_algebra", 
        "high_school_biology", 
        "high_school_world_history", 
        "marketing", 
        "philosophy", 
        "professional_law",
        "abstract_algebra_corpus",
        "anne_corpus",
        "college_computer_science_corpus",
        "food_corpus",
        "high_school_biology_corpus",
        "high_school_world_history_corpus",
        "marketing_corpus",
        "philosophy_corpus",
        "professional_law_corpus"
    ]
    
    print("="*70)
    print("PLOTTING LAYER PRUNING ANALYSIS")
    print("="*70)

    for model in models:
        for dataset in datasets:
            print(f"Processing Model: {model}, Dataset: {dataset}")
            base_prune_ratio = 0.0
            #plot_single_layer_impact_self(model, dataset, base_prune_ratio)
            plot_cosine_similarity(model, dataset)
            print()

if __name__ == '__main__':
    main()