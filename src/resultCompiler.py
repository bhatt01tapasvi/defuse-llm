from collections import defaultdict
import glob
import os 
import numpy as np
import pandas as pd
import json
import re
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from cmcrameri import cm

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
        #pattern = os.path.join(LAYER_ANALYSIS_DIR, f"{model_name}_{dataset_name}_single_*")
        pattern = os.path.join(LAYER_ANALYSIS_DIR, f"{model_name}_single_*")
    
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

    if "corpus" in dataset_name:
        dataset_name = f"custom_{dataset_name}"
    else:
        dataset_name = f"mmlu_{dataset_name}"

    base_dir = os.path.join(base_dir, dataset_name)

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
    dataset_plot_dir = os.path.join(PLOTS_DIR, "layer_analysis_v2", dataset_name, "single")
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
    plt.xlabel("Pruned Layer Index", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
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
    csv_path = "/users/grad/abhishektyagi/wanda/wanda/results/mlp_impact/meta-llama_Llama-3.2-3B/mmlu_marketing_mlp_impact.csv"
    
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
        plt.figure(figsize=(10, 5))
        plt.plot(layers, avg_cosine, marker='o', linewidth=2.5, markersize=8, 
                color='#1f77b4', label='Average', alpha=0.85, zorder=3)
        plt.plot(layers, min_cosine, marker='s', linewidth=2, markersize=6, 
                color='#d62728', label='Minimum', alpha=0.75, linestyle='--', zorder=2)
        plt.plot(layers, max_cosine, marker='^', linewidth=2, markersize=6, 
                color='#2ca02c', label='Maximum', alpha=0.75, linestyle='--', zorder=2)
        plt.fill_between(layers, min_cosine, max_cosine, alpha=0.15, color='gray', zorder=1)
        
        plt.xlabel('Layer Index', fontsize=18)
        plt.ylabel('Cosine Similarity', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.ylim([0, 1.05])
        plt.legend(frameon=True, loc='best', fontsize=18)
        plt.tight_layout()
        
        filename = f"cosine_similarity_{dataset_name}.pdf"
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

def rank_layers_no_bound(model_name: str, dataset_name: str, total_prune_percent: float = 50.0):
    """Rank layers and calculate pruning ratios to achieve total pruning target."""
    csv_path = find_mlp_impact_csv(model_name, dataset_name)
    
    if not csv_path:
        print(f"  ✗ No MLP impact CSV found for {dataset_name}")
        return
    
    print(f"  ✓ Found CSV: {os.path.basename(csv_path)}")
    
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Extract layer numbers and metrics
        layers = []
        avg_cosine = []
        avg_delta_norm = []
        avg_before_norm = []
        
        for _, row in df.iterrows():
            layer_str = row['layer']
            if isinstance(layer_str, str) and layer_str.startswith('layer_'):
                layer_num = int(layer_str.split('_')[1])
                layers.append(layer_num)
                avg_cosine.append(row['avg_cosine_sim'])
                avg_delta_norm.append(row['avg_delta_norm'])
                avg_before_norm.append(row['avg_before_norm'])

        if not layers:
            print(f"  ✗ No valid layer data found in CSV")
            return
        
        # Calculate scores: S = (1 - avg_cosine_sim) * (avg_delta_norm / avg_before_norm)
        # Higher score = more important = prune LESS
        layer_scores = []
        for i in range(len(layers)):
            score = (1 - avg_cosine[i]) * (avg_delta_norm[i] / avg_before_norm[i])
            layer_scores.append((layers[i], score))
        
        # Sort by score in descending order (highest score = most important)
        layer_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Print ranking
        print(f"\n  Layer Ranking (by score S = (1 - avg_cosine) * (delta_norm / before_norm)):")
        print(f"  {'Rank':<6} {'Layer':<8} {'Score':<12} {'Importance':<15}")
        print(f"  {'-'*50}")
        for rank, (layer, score) in enumerate(layer_scores, 1):
            importance = "High (prune less)" if rank <= len(layer_scores) // 3 else "Low (prune more)"
            print(f"  {rank:<6} {layer:<8} {score:<12.6f} {importance}")
        
        # Calculate pruning ratios to achieve target
        print(f"\n  {'='*60}")
        print(f"  Pruning Strategy to achieve {total_prune_percent}% total pruning")
        print(f"  {'='*60}")
        
        num_layers = len(layers)
        total_neurons = num_layers  # Assuming each layer has same size, normalized to 1
        target_prune_amount = total_prune_percent / 100.0 * total_neurons
        
        # Invert scores: lower original score = higher pruning weight
        # Calculate inverse scores for pruning distribution
        max_score = max(score for _, score in layer_scores)
        inverse_scores = [(layer, max_score - score + 1e-6) for layer, score in layer_scores]
        total_inverse_score = sum(inv_score for _, inv_score in inverse_scores)
        
        # Calculate pruning ratio for each layer inversely proportional to its importance score
        pruning_plan = []
        actual_total_pruned = 0
        
        for (layer, orig_score), (_, inv_score) in zip(layer_scores, inverse_scores):
            # Prune proportionally to inverse score (low importance = high pruning)
            layer_prune_ratio = (inv_score / total_inverse_score) * target_prune_amount / 1.0
            # Cap at 100% per layer
            layer_prune_ratio = min(layer_prune_ratio, 1.0)
            layer_prune_percent = layer_prune_ratio * 100
            
            pruning_plan.append((layer, orig_score, layer_prune_ratio, layer_prune_percent))
            actual_total_pruned += layer_prune_ratio
        
        # Sort pruning plan by layer number for display
        pruning_plan_by_layer = sorted(pruning_plan, key=lambda x: x[0])
        
        # Print pruning plan
        print(f"\n  {'Layer':<8} {'Score':<12} {'Prune Ratio':<15} {'Prune %':<12} {'Keep %':<12}")
        print(f"  {'-'*65}")
        for layer, score, ratio, percent in pruning_plan_by_layer:
            keep_percent = 100 - percent
            print(f"  {layer:<8} {score:<12.6f} {ratio:<15.4f} {percent:<12.2f} {keep_percent:<12.2f}")
        
        actual_total_percent = (actual_total_pruned / num_layers) * 100
        print(f"\n  Target Total Pruning: {total_prune_percent}%")
        print(f"  Actual Total Pruning: {actual_total_percent:.2f}%")
        print(f"  {'='*60}\n")
        
    except Exception as e:
        print(f"  ✗ Error ranking: {e}")

def rank_layers(model_name: str, dataset_name: str, total_prune_percent: float = 50.0,
                we: float = 0.3, wl: float = 0.15, alpha: float = 0.25, beta: float = 0.35,
                p_min: float = 0.10, p_max: float = 0.90, epsilon: float = 1e-6,
                max_iterations: int = 10):
    """
    Rank layers and calculate pruning ratios using depth-aware protection.
    
    Implements the 9-step pruning strategy:
    1. Calculate per-layer functional importance S_ℓ
    2. Normalize importance to [0,1] → Ŝ_ℓ
    3. Calculate relative pruning pressure R_ℓ = 1 - Ŝ_ℓ
    4. Apply depth-aware protection factor D_ℓ
    5. Compute final pruning pressure P_ℓ = R_ℓ · D_ℓ
    6. Allocate pruning budget proportionally → p̃_ℓ
    7. Enforce safety bounds [p_min, p_max] → p_ℓ^(0)
    8. Iteratively redistribute residual budget Δ
    9. Return final pruning ratios satisfying constraints
    
    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        total_prune_percent: Target total pruning percentage ρ (e.g., 50.0 for 50%)
        we: Early protection width (default: 0.2)
        wl: Late protection width (default: 0.15)
        alpha: Early minimum scaling factor (default: 0.25)
        beta: Late minimum scaling factor (default: 0.35)
        p_min: Minimum pruning ratio per layer (default: 0.10)
        p_max: Maximum pruning ratio per layer (default: 0.90)
        epsilon: Small constant for numerical stability (default: 1e-6)
        max_iterations: Maximum iterations for budget correction (default: 100)
    """
    csv_path = find_mlp_impact_csv(model_name, dataset_name)
    
    if not csv_path:
        print(f"  ✗ No MLP impact CSV found for {dataset_name}")
        return
    
    print(f"  ✓ Found CSV: {os.path.basename(csv_path)}")
    
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Extract layer numbers and metrics
        layers = []
        avg_cosine = []
        avg_delta_norm = []
        avg_before_norm = []
        
        for _, row in df.iterrows():
            layer_str = row['layer']
            if isinstance(layer_str, str) and layer_str.startswith('layer_'):
                layer_num = int(layer_str.split('_')[1])
                layers.append(layer_num)
                avg_cosine.append(row['avg_cosine_sim'])
                avg_delta_norm.append(row['avg_delta_norm'])
                avg_before_norm.append(row['avg_before_norm'])

        if not layers:
            print(f"  ✗ No valid layer data found in CSV")
            return
        
        layers = np.array(layers)
        avg_cosine = np.array(avg_cosine)
        avg_delta_norm = np.array(avg_delta_norm)
        avg_before_norm = np.array(avg_before_norm)
        
        num_layers = len(layers)
        L = num_layers
        rho = total_prune_percent / 100.0
        B = rho * L  # Total pruning budget
        
        print(f"\n  {'='*80}")
        print(f"  Depth-Aware Pruning Strategy")
        print(f"  {'='*80}")
        print(f"  Target: {total_prune_percent}% total pruning (B = ρL = {B:.2f})")
        print(f"  Layers: L = {L}")
        print(f"  Config: we={we}, wl={wl}, α={alpha}, β={beta}, p_min={p_min}, p_max={p_max}")
        
        # ==================== STEP 1: Per-layer functional importance ====================
        # S_ℓ = (1 - cos_ℓ) · ||Δx_ℓ|| / ||x_ℓ,pre||
        S = (1 - avg_cosine) * (avg_delta_norm / (avg_before_norm + epsilon))
        
        print(f"\n  Step 1: Functional importance S_ℓ calculated")
        print(f"    Range: [{np.min(S):.6f}, {np.max(S):.6f}]")
        
        # ==================== STEP 2: Normalize importance ====================
        # Ŝ_ℓ = (S_ℓ - min S_j) / (max S_j - min S_j + ε)
        S_min = np.min(S)
        S_max = np.max(S)
        S_hat = (S - S_min) / (S_max - S_min + epsilon)
        
        print(f"  Step 2: Normalized importance Ŝ_ℓ ∈ [0,1]")
        print(f"    Range: [{np.min(S_hat):.6f}, {np.max(S_hat):.6f}]")
        
        # ==================== STEP 3: Relative pruning pressure ====================
        # R_ℓ = 1 - Ŝ_ℓ
        R = 1 - S_hat
        
        print(f"  Step 3: Pruning pressure R_ℓ = 1 - Ŝ_ℓ")
        print(f"    Range: [{np.min(R):.6f}, {np.max(R):.6f}]")
        
        # ==================== STEP 4: Depth-aware protection factor ====================
        # z_ℓ = ℓ / (L - 1)
        z = layers / (L - 1)
        D = np.ones(num_layers)
        
        for i in range(num_layers):
            if z[i] < we:
                # Early layers: α + (1 - α) * z_ℓ / w_e
                D[i] = alpha + (1 - alpha) * (z[i] / we)
            elif z[i] > (1 - wl):
                # Late layers: β + (1 - β) * (1 - z_ℓ) / w_l
                D[i] = beta + (1 - beta) * ((1 - z[i]) / wl)
            else:
                # Middle layers: no protection
                D[i] = 1.0
        
        print(f"  Step 4: Depth protection D_ℓ applied")
        print(f"    Range: [{np.min(D):.6f}, {np.max(D):.6f}]")
        print(f"    Early protected: {np.sum(z < we)} layers")
        print(f"    Late protected:  {np.sum(z > (1 - wl))} layers")
        
        # ==================== STEP 5: Final pruning pressure ====================
        # P_ℓ = R_ℓ · D_ℓ
        P_pressure = R * D
        
        print(f"  Step 5: Final pressure P_ℓ = R_ℓ · D_ℓ")
        print(f"    Range: [{np.min(P_pressure):.6f}, {np.max(P_pressure):.6f}]")
        
        # ==================== STEP 6: Budgeted pruning allocation ====================
        # p̃_ℓ = B · P_ℓ / Σ_j P_j
        P_sum = np.sum(P_pressure)
        p_tilde = B * (P_pressure / P_sum)
        
        print(f"  Step 6: Initial allocation p̃_ℓ (before bounds)")
        print(f"    Range: [{np.min(p_tilde):.6f}, {np.max(p_tilde):.6f}]")
        print(f"    Sum: {np.sum(p_tilde):.6f} (target: {B:.6f})")
        
        # ==================== STEP 7: Enforce safety bounds ====================
        # p_ℓ^(0) = clip(p̃_ℓ, p_min, p_max)
        p_0 = np.clip(p_tilde, p_min, p_max)
        
        initial_sum = np.sum(p_0)
        initial_delta = B - initial_sum
        
        print(f"  Step 7: Applied bounds [p_min={p_min}, p_max={p_max}]")
        print(f"    Sum after clipping: {initial_sum:.6f}")
        print(f"    Initial residual Δ: {initial_delta:.6f}")
        
        # ==================== STEP 8: Budget correction ====================
        # Iteratively redistribute Δ among layers not at p_max (when Δ > 0)
        p = p_0.copy()
        iteration = 0
        
        print(f"\n  Step 8: Budget redistribution (iterative)")
        
        while iteration < max_iterations:
            current_sum = np.sum(p)
            delta = B - current_sum
            
            # Check convergence
            if abs(delta) < epsilon:
                print(f"    ✓ Converged at iteration {iteration}: |Δ| = {abs(delta):.9f} < ε")
                break
            
            # Determine which layers can be adjusted based on sign of delta
            if delta > 0:
                # Need to prune MORE: only adjust layers below p_max (can increase pruning)
                available_mask = p < (p_max - epsilon)
                operation = "increase pruning"
            else:
                # Need to prune LESS: only adjust layers above p_min (can decrease pruning)
                available_mask = p > (p_min + epsilon)
                operation = "decrease pruning"
            
            if not np.any(available_mask):
                if delta > 0:
                    print(f"    ⚠ All layers at p_max. Cannot prune more.")
                else:
                    print(f"    ⚠ All layers at p_min. Cannot prune less.")
                print(f"      Final sum: {current_sum:.6f}, Target: {B:.6f}, Gap: {delta:.6f}")
                break
            
            # Weighted redistribution: p_ℓ = p_ℓ + Δ · [indicator · P_ℓ] / Σ[indicator · P_j]
            available_P = P_pressure * available_mask
            P_available_sum = np.sum(available_P)
            
            if P_available_sum > epsilon:
                redistribution = delta * (available_P / P_available_sum)
                p = p + redistribution
                
                # Re-clip to bounds
                p = np.clip(p, p_min, p_max)
            else:
                # Fallback: uniform redistribution if no pressure weights available
                num_available = np.sum(available_mask)
                redistribution = delta / num_available
                p[available_mask] += redistribution
                p = np.clip(p, p_min, p_max)
            
            iteration += 1
            
            if iteration % 10 == 0:
                print(f"    Iteration {iteration}: Δ = {delta:.9f}, {operation}, available = {np.sum(available_mask)}")
        
        if iteration >= max_iterations:
            print(f"    ⚠ Reached max iterations ({max_iterations})")
        
        # ==================== STEP 9: Final results ====================
        final_sum = np.sum(p)
        final_percent = (final_sum / L) * 100
        
        print(f"\n  Step 9: Final pruning ratios")
        print(f"    Sum: {final_sum:.6f} / {B:.6f} (satisfaction: {(final_sum/B)*100:.2f}%)")
        print(f"    Average per-layer: {final_percent:.2f}%")
        
        # Convert to percentages for display
        p_percent = p * 100
        keep_percent = 100 - p_percent
        
        # ==================== Detailed Output ====================
        print(f"\n  {'='*115}")
        print(f"  Layer-wise Breakdown")
        print(f"  {'='*115}")
        print(f"  {'Layer':<6} {'z_ℓ':<8} {'S_ℓ':<10} {'Ŝ_ℓ':<10} {'R_ℓ':<10} {'D_ℓ':<10} {'P_ℓ':<10} {'p̃_ℓ':<10} {'p_ℓ':<10} {'Prune%':<10} {'Keep%':<10}")
        print(f"  {'-'*115}")
        
        for i in range(num_layers):
            print(f"  {layers[i]:<6} {z[i]:<8.4f} {S[i]:<10.6f} {S_hat[i]:<10.6f} {R[i]:<10.6f} {D[i]:<10.6f} "
                  f"{P_pressure[i]:<10.6f} {p_tilde[i]:<10.6f} {p[i]:<10.6f} {p_percent[i]:<10.2f} {keep_percent[i]:<10.2f}")
        
        # ==================== Summary Statistics ====================
        print(f"\n  {'='*80}")
        print(f"  Summary Statistics")
        print(f"  {'='*80}")
        print(f"  Target Total Pruning:     {total_prune_percent:.2f}%")
        print(f"  Actual Total Pruning:     {final_percent:.2f}%")
        print(f"  Budget Satisfaction:      {(final_sum / B) * 100:.4f}%")
        print(f"  Min Layer Pruning:        {np.min(p_percent):.2f}% (layer {layers[np.argmin(p_percent)]})")
        print(f"  Max Layer Pruning:        {np.max(p_percent):.2f}% (layer {layers[np.argmax(p_percent)]})")
        print(f"  Mean Layer Pruning:       {np.mean(p_percent):.2f}%")
        print(f"  Std Dev:                  {np.std(p_percent):.2f}%")
        print(f"  Layers at p_min ({p_min}):     {np.sum(np.abs(p - p_min) < epsilon)}")
        print(f"  Layers at p_max ({p_max}):     {np.sum(np.abs(p - p_max) < epsilon)}")
        
        # ==================== Top/Bottom Layers ====================
        sorted_indices_by_importance = np.argsort(S)[::-1]
        sorted_indices_by_pruning = np.argsort(p_percent)
        
        print(f"\n  Top 5 Most Important Layers (highest S_ℓ, should prune less):")
        for idx in sorted_indices_by_importance[:min(5, num_layers)]:
            print(f"    Layer {layers[idx]:2d}: S={S[idx]:.6f}, Ŝ={S_hat[idx]:.4f}, "
                  f"D={D[idx]:.4f}, prune={p_percent[idx]:5.2f}%")
        
        print(f"\n  Top 5 Least Important Layers (lowest S_ℓ, should prune more):")
        for idx in sorted_indices_by_importance[-min(5, num_layers):]:
            print(f"    Layer {layers[idx]:2d}: S={S[idx]:.6f}, Ŝ={S_hat[idx]:.4f}, "
                  f"D={D[idx]:.4f}, prune={p_percent[idx]:5.2f}%")
        
        print(f"\n  Top 5 Most Pruned Layers:")
        for idx in sorted_indices_by_pruning[-min(5, num_layers):]:
            print(f"    Layer {layers[idx]:2d}: prune={p_percent[idx]:5.2f}%, "
                  f"S={S[idx]:.6f}, D={D[idx]:.4f}")
        
        print(f"\n  Top 5 Least Pruned Layers:")
        for idx in sorted_indices_by_pruning[:min(5, num_layers)]:
            print(f"    Layer {layers[idx]:2d}: prune={p_percent[idx]:5.2f}%, "
                  f"S={S[idx]:.6f}, D={D[idx]:.4f}")
        
        print(f"  {'='*80}\n")
        
        # Return results as dictionary
        return {
            'layers': layers,
            'scores': S,
            'normalized_scores': S_hat,
            'pruning_pressure': P_pressure,
            'pruning_ratios': p,
            'pruning_percentages': p_percent,
            'depth_factors': D,
            'relative_pressure': R,
            'depth_positions': z,
            'initial_allocation': p_tilde,
            'budget_target': B,
            'budget_achieved': final_sum
        }
        
    except Exception as e:
        print(f"  ✗ Error ranking: {e}")
        import traceback
        traceback.print_exc()
        return None

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

    datasets_01 = [
        "college_computer_science", 
        "abstract_algebra", 
        "high_school_biology", 
        "marketing", 
        "abstract_algebra_corpus",
        "college_computer_science_corpus",
        "high_school_biology_corpus",
        "marketing_corpus",
    ]

    dataset_2 = ["college_computer_science", "marketing"]
    
    print("="*70)
    print("PLOTTING LAYER PRUNING ANALYSIS")
    print("="*70)

    for model in models:
        for dataset in dataset_2:
            print(f"Processing Model: {model}, Dataset: {dataset}")
            base_prune_ratio = 0.0
            plot_single_layer_impact_self(model, dataset, base_prune_ratio)
            #plot_cosine_similarity(model, dataset)
            #rank_layers(model, dataset)
            print()

def main_temp():
    plot_cosine_similarity(model_name="yp", dataset_name="MMLU_MMLU")

if __name__ == '__main__':
    main_temp()
