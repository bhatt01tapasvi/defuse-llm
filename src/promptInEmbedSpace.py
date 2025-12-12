import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
from util import data_loader
from scipy.spatial.distance import cosine
import pandas as pd

# Global path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DOMAIN_CONSISTENCY_DIR = os.path.join(RESULTS_DIR, "domain_consistency")
MLP_IMPACT_DIR = os.path.join(RESULTS_DIR, "mlp_impact")

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DOMAIN_CONSISTENCY_DIR, exist_ok=True)
os.makedirs(MLP_IMPACT_DIR, exist_ok=True)

global_means = {}

def compute_global_means(all_datasets: list[Tuple[str, Dict[str, Any]]], embed: str):
    # Global embedding accumulation across all prompts
    print("\n" + "="*80)
    print("COMPUTING GLOBAL EMBEDDINGS ACROSS ALL PROMPTS")
    print("="*80)
    
    global global_means
    global_token_sums = defaultdict(lambda: np.zeros(768))  # Assuming 768 dim for GPT-2
    global_token_counts = defaultdict(int)
    
    for dataset_name, dataset in all_datasets:    
        # Get the last generation for each layer
        for layer_name, activations_list in dataset[embed].items():
            last_gen_activations = activations_list[-1]  # Shape: (num_tokens, embed_dim)
            
            # Sum all tokens in this layer for this dataset
            for token_idx in range(last_gen_activations.shape[0]):
                token_embedding = last_gen_activations[token_idx]  # Shape: (embed_dim,)
                global_token_sums[layer_name] += token_embedding
                global_token_counts[layer_name] += 1
    
    # Compute global means
    for layer_name in global_token_sums.keys():
        global_means[layer_name] = global_token_sums[layer_name] / global_token_counts[layer_name]
    
def euclidean_distance(emb1, emb2):
    """Sensitive to magnitude - good for normalized embeddings"""
    emb1 = emb1.astype(np.float64)
    emb2 = emb2.astype(np.float64)
    distance = np.linalg.norm(emb1 - emb2)
    # Handle potential inf or very large values
    if np.isinf(distance):
        return float('inf')
    elif np.isnan(distance):
        return 0.0  # If both are NaN, consider distance as 0
    else:
        return distance

def cosine_similarity(emb1, emb2):
    """Most popular for embeddings - scale-invariant"""
    try:
        emb1 = emb1.astype(np.float64)
        emb2 = emb2.astype(np.float64)
        distance = cosine(emb1, emb2)
        # Handle edge cases that scipy returns
        if np.isnan(distance):
            # Both are zero vectors
            print("Both vectors are zero vectors, returning similarity of 1.0")
            return 1.0
        
        similarity = 1.0 - distance
        return similarity
    except:
        # Fallback for any errors
        print("Error computing cosine similarity, checking for zero vectors.")
        return 1.0 if np.allclose(emb1, emb2) else 0.0

def compareInEmbedSpace(activations_a: Dict[str, list], activations_b: Dict[str, list]):
    # Here we are trying to understand how each token is populated with the global context of the sentence.
    # We particularly focus on the last token of each generation step.
    # Bear in mind that we shouldn't stride across generation steps and just pick the last token because the very first generation already has pre-fill token available.
    # For the decode phase, each new token generated is the last token which we need to analyze.
    # Easy approach would be to literally, just take the last generation step and study each token of it.

    last_gen_a = {layer: activations[-1] for layer, activations in activations_a.items()}
    last_gen_b = {layer: activations[-1] for layer, activations in activations_b.items()}
    sentence_embed_a = {}
    sentence_embed_b = {}
    lasttokenwise_comparisons = defaultdict(dict)
    sentencewise_comparisons = defaultdict(dict)

    for layer in last_gen_a.keys():
        activations_layer_a = last_gen_a[layer]  # Shape: (num_tokens, embed_dim)
        activations_layer_b = last_gen_b[layer]  # Shape: (num_tokens, embed_dim)
        
        # Initialize sentence embeddings for this layer
        sentence_embed_a[layer] = np.zeros_like(activations_layer_a[0])
        sentence_embed_b[layer] = np.zeros_like(activations_layer_b[0])

        for token_idx in range(activations_layer_a.shape[0]):
            token_a = activations_layer_a[token_idx]  # Shape: (embed_dim,)
            sentence_embed_a[layer] += token_a

        for token_idx in range(activations_layer_b.shape[0]):
            token_b = activations_layer_b[token_idx]  # Shape: (embed_dim,)
            sentence_embed_b[layer] += token_b
        
       
        sentence_embed_a[layer] /= activations_layer_a.shape[0]
        sentence_embed_b[layer] /= activations_layer_b.shape[0]
        #sentence_embed_mean = np.mean([sentence_embed_a[layer], sentence_embed_b[layer]], axis=1)
        
        #sentence_embed_a[layer] = sentence_embed_a[layer] - sentence_embed_mean
        #sentence_embed_b[layer] = sentence_embed_b[layer] - sentence_embed_mean
        sentence_sim = cosine_similarity(sentence_embed_a[layer] - global_means[layer], sentence_embed_b[layer] - global_means[layer])
        sentence_euclidean = euclidean_distance(sentence_embed_a[layer] - global_means[layer], sentence_embed_b[layer] - global_means[layer])
        sentencewise_comparisons[layer]['similarity'] = sentence_sim
        sentencewise_comparisons[layer]['distance'] = sentence_euclidean
        sent_mag_ratio = np.linalg.norm(sentence_embed_a[layer] - global_means[layer]) / (np.linalg.norm(sentence_embed_b[layer] - global_means[layer]) + 1e-8)
        sentencewise_comparisons[layer]['mag_ratio'] = sent_mag_ratio
        sent_top_100_a = set(np.argsort(np.abs(sentence_embed_a[layer] - global_means[layer]))[-100:])
        sent_top_100_b = set(np.argsort(np.abs(sentence_embed_b[layer] - global_means[layer]))[-100:])
        sentencewise_comparisons[layer]['jaccard'] = len(sent_top_100_a & sent_top_100_b) / len(sent_top_100_a | sent_top_100_b)

        lasttokenwise_comparisons[layer]['similarity'] = cosine_similarity(activations_layer_a[-1] - global_means[layer], activations_layer_b[-1] - global_means[layer])
        lasttokenwise_comparisons[layer]['distance'] = euclidean_distance(activations_layer_a[-1] - global_means[layer], activations_layer_b[-1] - global_means[layer])
        last_tok_mag_ratio = np.linalg.norm(activations_layer_a[-1] - global_means[layer]) / (np.linalg.norm(activations_layer_b[-1] - global_means[layer]) + 1e-8)
        lasttokenwise_comparisons[layer]['mag_ratio'] = last_tok_mag_ratio
        last_tok_top_100_a = set(np.argsort(np.abs(activations_layer_a[-1] - global_means[layer]))[-100:])
        last_tok_top_100_b = set(np.argsort(np.abs(activations_layer_b[-1] - global_means[layer]))[-100:])
        lasttokenwise_comparisons[layer]['jaccard'] = len(last_tok_top_100_a & last_tok_top_100_b) / len(last_tok_top_100_a | last_tok_top_100_b)
    return lasttokenwise_comparisons, sentencewise_comparisons

def create_comparison_table(token_comp, sent_comp, path_to_save, comparison_name):
    """Create a formatted table for comparison results"""
    # Extract layer names and sort them numerically instead of alphabetically
    def extract_block_number(layer_name):
        # Extract the number from 'block_X' format
        if 'block_' in layer_name:
            return int(layer_name.split('_')[1])
        return 0  # fallback for any unexpected format
    
    layers = sorted(token_comp.keys(), key=extract_block_number)
    
    # Create data for the table
    table_data = []
    for layer in layers:
        row = {
            'Layer': layer,
            'Sentence Similarity': f"{sent_comp[layer]['similarity']:.4f}",
            'Sentence Euclidean Dist': f"{sent_comp[layer]['distance']:.2f}",
            'Sentence Mag Ratio': f"{sent_comp[layer]['mag_ratio']:.4f}",
            'Sentence Jaccard': f"{sent_comp[layer]['jaccard']:.4f}",
            'Token Similarity': f"{token_comp[layer]['similarity']:.4f}",
            'Token Euclidean Dist': f"{token_comp[layer]['distance']:.2f}",
            'Token Mag Ratio': f"{token_comp[layer]['mag_ratio']:.4f}",
            'Token Jaccard': f"{token_comp[layer]['jaccard']:.4f}"
        }
        table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Print formatted table
    print(f"\n{comparison_name}")
    print("=" * len(comparison_name))
    
    # Save to CSV
    df.to_csv(path_to_save, index=False)
    print(f"Table saved to: {path_to_save}")
    print(df.to_string(index=False, justify='center'))
    print("\n")

def create_consistency_table(token_comp_list, sent_comp_list, path_to_save, comparison_name):
    """Create a formatted table for consistency results"""
    # Create data for the table
    table_data = []
    for i in range(len(token_comp_list)):
        # Get the last layer (since we're only working with one layer in domain consistency)
        last_layer = list(token_comp_list[i].keys())[0]
        row = {
            'Num Tokens': i + 1,
            'Sentence Similarity': f"{sent_comp_list[i][last_layer]['similarity']:.4f}",
            'Sentence Euclidean Dist': f"{sent_comp_list[i][last_layer]['distance']:.2f}",
            'Sentence Mag Ratio': f"{sent_comp_list[i][last_layer]['mag_ratio']:.4f}",
            'Sentence Jaccard': f"{sent_comp_list[i][last_layer]['jaccard']:.4f}",
            'Token Similarity': f"{token_comp_list[i][last_layer]['similarity']:.4f}",
            'Token Euclidean Dist': f"{token_comp_list[i][last_layer]['distance']:.2f}",
            'Token Mag Ratio': f"{token_comp_list[i][last_layer]['mag_ratio']:.4f}",
            'Token Jaccard': f"{token_comp_list[i][last_layer]['jaccard']:.4f}"
        }
        table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Print formatted table
    print(f"\n{comparison_name}")
    print("=" * len(comparison_name))
    
    # Save to CSV
    df.to_csv(path_to_save, index=False)
    print(f"Table saved to: {path_to_save}")
    print(df.to_string(index=False, justify='center'))
    print("\n")

def compareInEmbedSpaceSummary(datasets: list[Tuple[str, Any]], embed: str):

    # Compare each pair of datasets only once (avoid duplicates like A vs B and B vs A)
    for i, (name_a, data_a) in enumerate(datasets):
        for j, (name_b, data_b) in enumerate(datasets):
            if i < j:  # Only compare when i < j to avoid duplicates
                token_comp, sent_comp = compareInEmbedSpace(data_a[embed], data_b[embed])
                save_path = os.path.join(RESULTS_DIR, f"{name_a}_vs_{name_b}_comparison.csv")
                create_comparison_table(token_comp, sent_comp, save_path, f"{name_a} vs {name_b} Comparison")

def compareDomainConsistency(data_a: Dict[str, Any], data_b: Dict[str, Any], embed: str):

    #We will only study the last layer because that determines how much the domain consistency is maintained.
    last_layer = list(data_a[embed].items())[-1][0]
    
    # Maintain the dictionary structure that compareInEmbedSpace expects
    # Structure: {layer_name: [forward_passes]} where each forward_pass is [tokens][dimensions]
    modified_data_a = {last_layer: data_a[embed][last_layer]}
    
    # Compute domain consistency metrics
    # I want to keep the first embed constant but keep sending progressive tokens of the second data vector.
    num_tokens = data_b[embed][last_layer][-1].shape[0]
    token_comp_list = []
    sent_comp_list = []

    for i in range(num_tokens):
        # Take the first i+1 tokens from the last forward pass
        truncated_tokens = data_b[embed][last_layer][-1][:i+1]

        # Maintain the list structure: [forward_passes] where each is [tokens][dimensions]
        modified_data_b = {last_layer: [truncated_tokens]}
        token_comp, sent_comp = compareInEmbedSpace(modified_data_a, modified_data_b)
        token_comp_list.append(token_comp)
        sent_comp_list.append(sent_comp)

    save_path = os.path.join(DOMAIN_CONSISTENCY_DIR, "domain_consistency_analysis.csv")
    create_consistency_table(token_comp_list, sent_comp_list, save_path, f"Domain Consistency Table")

def create_comparison_table_mlp(results: list, path_to_save: str, dataset_name: str):
    """Create a formatted table for MLP impact results"""
    
    # Extract layer names and sort them numerically
    def extract_block_number(layer_name):
        if 'block_' in layer_name:
            return int(layer_name.split('_')[1])
        return 0
    
    # Sort results by block number
    sorted_results = sorted(results, key=lambda x: extract_block_number(x['layer']))
    
    # Create DataFrame
    df = pd.DataFrame(sorted_results)
    
    # Print formatted table
    print(f"\n{dataset_name} - MLP Residual Impact Analysis")
    print("=" * 100)
    print("Comparing: Residual_Before vs Residual_After = Residual_Before + MLP_Output")
    print("=" * 100)
    print(df.to_string(index=False, justify='center'))
    print("\n")
    
    # Save to CSV
    df.to_csv(path_to_save, index=False)
    print(f"Table saved to: {path_to_save}\n")
    
    return df

def compare_activation_states(
    activations_before: Dict[str, list], 
    activations_after: Dict[str, list],
    comparison_name: str = "Activation Comparison"
):
    """
    General function to compare two activation states.
    """
    
    results = []
    
    for layer_name in activations_before.keys():
        # ===== CONVERT TO FLOAT64 FIRST! =====
        before_float16 = activations_before[layer_name][-1]  # Original float16
        after_float16 = activations_after[layer_name][-1]    # Original float16
        
        # Check for inf/nan in ORIGINAL data (before conversion)
        before_has_inf = np.any(np.isinf(before_float16))
        after_has_inf = np.any(np.isinf(after_float16))
        
        if before_has_inf or after_has_inf:
            print(f"\n⚠️  Float16 overflow detected in layer {layer_name}:")
            if before_has_inf:
                num_inf = np.sum(np.isinf(before_float16))
                total_elements = before_float16.size
                pct_inf = (num_inf / total_elements) * 100
                max_finite = np.max(before_float16[np.isfinite(before_float16)])
                print(f"    'before': {num_inf}/{total_elements} values are inf ({pct_inf:.1f}%)")
                print(f"    Max finite value: {max_finite:.2e}")
            if after_has_inf:
                num_inf = np.sum(np.isinf(after_float16))
                total_elements = after_float16.size
                pct_inf = (num_inf / total_elements) * 100
                max_finite = np.max(after_float16[np.isfinite(after_float16)])
                print(f"    'after': {num_inf}/{total_elements} values are inf ({pct_inf:.1f}%)")
                print(f"    Max finite value: {max_finite:.2e}")
        
        # NOW convert to float64 (this won't fix existing inf, but prevents new overflow)
        before = before_float16.astype(np.float64)  # Shape: (num_tokens, embed_dim)
        after = after_float16.astype(np.float64)    # Shape: (num_tokens, embed_dim)
        
        num_tokens = before.shape[0]
        
        # Token-wise analysis
        token_similarities = []
        token_distances = []
        token_relative_changes = []
        token_delta_norms = []
        token_before_norms = []
        token_after_norms = []
        
        skipped_tokens = 0
        
        for token_idx in range(num_tokens):
            tok_before = before[token_idx]  # Already float64
            tok_after = after[token_idx]    # Already float64
            
            # Check if THIS TOKEN has inf/nan (from original float16 overflow)
            has_inf_before = np.any(np.isinf(tok_before)) or np.any(np.isnan(tok_before))
            has_inf_after = np.any(np.isinf(tok_after)) or np.any(np.isnan(tok_after))
            
            if has_inf_before or has_inf_after:
                skipped_tokens += 1
                continue  # Skip this token
            
            # Compute delta (what changed)
            delta = tok_after - tok_before
            
            # Compute norms (safe now, no inf in inputs)
            before_norm = np.linalg.norm(tok_before)
            after_norm = np.linalg.norm(tok_after)
            delta_norm = np.linalg.norm(delta)
            
            # Sanity check: if norm computation overflows in float64 (very rare)
            if np.isinf(delta_norm) or np.isinf(before_norm) or np.isinf(after_norm):
                print(f"⚠️  Norm overflow in float64 at layer {layer_name}, token {token_idx}")
                print(f"    This is unusual - check data integrity!")
                skipped_tokens += 1
                continue
            
            # Metrics
            cos_sim = cosine_similarity(tok_before, tok_after)
            eucl_dist = euclidean_distance(tok_before, tok_after)
            
            # Relative change
            if before_norm < 1e-8:
                relative_change = 0.0 if delta_norm < 1e-8 else 1e6
            else:
                relative_change = delta_norm / before_norm
                relative_change = min(relative_change, 1e6)
            
            token_similarities.append(cos_sim)
            token_distances.append(eucl_dist)
            token_relative_changes.append(relative_change)
            token_delta_norms.append(delta_norm)
            token_before_norms.append(before_norm)
            token_after_norms.append(after_norm)
        
        if skipped_tokens > 0:
            print(f"    Skipped {skipped_tokens}/{num_tokens} tokens due to inf/nan")
        
        # Safe aggregation
        def safe_mean(values):
            if not values:
                return 0.0
            finite_values = [v for v in values if np.isfinite(v)]
            return np.mean(finite_values) if finite_values else 0.0
        
        def safe_max(values):
            if not values:
                return 0.0
            finite_values = [v for v in values if np.isfinite(v)]
            return np.max(finite_values) if finite_values else 0.0
        
        def safe_min(values):
            if not values:
                return 0.0
            finite_values = [v for v in values if np.isfinite(v)]
            return np.min(finite_values) if finite_values else 0.0
        
        # Aggregate for this layer
        layer_result = {
            'layer': layer_name,
            'avg_cosine_sim': safe_mean(token_similarities),
            'min_cosine_sim': safe_min(token_similarities),
            'max_cosine_sim': safe_max(token_similarities),
            'avg_euclidean': safe_mean(token_distances),
            'avg_relative_change': safe_mean(token_relative_changes),
            'max_relative_change': safe_max(token_relative_changes),
            'avg_delta_norm': safe_mean(token_delta_norms),
            'avg_before_norm': safe_mean(token_before_norms),
            'avg_after_norm': safe_mean(token_after_norms),
            'num_valid_tokens': len(token_similarities),
            'num_total_tokens': num_tokens,
            'num_skipped_tokens': skipped_tokens,
        }
        
        results.append(layer_result)
    
    return results


def compareMLPImpact(datasets: list[Tuple[str, Any]], model_name: str = "unknown_model"):
    """
    Analyze MLP impact by comparing residual stream before and after MLP.
    
    In transformer: residual_after = residual_before + MLP(residual_before)
    
    We compare:
    - Before: pre_ln2_activations (residual stream entering MLP block)
    - After: post_layer_activations (residual stream after MLP added)
    
    Args:
        datasets: List of (dataset_name, data) tuples
        model_name: Name of the model being analyzed (creates subdirectory)
    """
    
    # Create model-specific subdirectory
    model_mlp_dir = os.path.join(MLP_IMPACT_DIR, model_name)
    os.makedirs(model_mlp_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("MLP RESIDUAL STREAM IMPACT ANALYSIS")
    print(f"Model: {model_name}")
    print("Comparing: Pre-MLP Residual vs Post-Layer Residual")
    print("="*80)
    print(f"Results will be saved to: {model_mlp_dir}\n")
    
    all_results = {}
    
    for dataset_name, data in datasets:
        print(f"\nAnalyzing dataset: {dataset_name}")
        
        # Verify data consistency (optional but good practice)
        is_consistent = True
        for layer_name in data['pre_ln2_activations'].keys():
            pre_ln2 = data['pre_ln2_activations'][layer_name][-1]
            post_mlp2 = data['post_mlp2_activations'][layer_name][-1]
            post_layer = data['post_layer_activations'][layer_name][-1]
            
            # Check if: post_layer ≈ pre_ln2 + post_mlp2
            expected = pre_ln2 + post_mlp2
            
            if not np.allclose(expected, post_layer, rtol=1e-5, atol=1e-6):
                print(f"⚠️  Data inconsistency in layer {layer_name}")
                print(f"    Max difference: {np.max(np.abs(expected - post_layer)):.6f}")
                is_consistent = False
                break
        
        if not is_consistent:
            print(f"❌ Skipping {dataset_name} due to inconsistency.\n")
            continue
        
        print(f"✓ Data consistency verified")
        
        # Compare pre-MLP vs post-layer (which includes MLP contribution)
        results = compare_activation_states(
            activations_before=data['pre_ln2_activations'],
            activations_after=data['post_layer_activations'],
            comparison_name=f"{dataset_name} MLP Impact"
        )
        
        # Save results to model-specific directory
        save_path = os.path.join(model_mlp_dir, f"{dataset_name}_mlp_impact.csv")
        df = create_comparison_table_mlp(results, save_path, dataset_name)
        
        all_results[dataset_name] = df
    
    return all_results

def structure_explainer(data: dict[str, Any]=None):
    if data is None:
        print("No data available for structure explanation.")
        return
    
    print("\n" + "="*60)
    print("OVERVIEW OF ALL ACTIVATION TYPES:")
    print("="*60)

    for activation_type, activation_dict in data.items():
        print(f"\n{activation_type}:")
        print(f"  Number of layers: {len(activation_dict)}")
        if len(activation_dict) > 0:
            # Get the first layer's data as a sample
            first_layer_key = list(activation_dict.keys())[0]
            first_layer_data = activation_dict[first_layer_key]
            print(f"  Sample layer: {first_layer_key}")
            print(f"  Number of forward passes in sample: {len(first_layer_data)}")
            if len(first_layer_data) > 0:
                print(f"  Final activation shape in sample: {first_layer_data[-1].shape}")
        
        # Show a few layer names to understand the structure
        layer_names = list(activation_dict.keys())[:3]
        print(f"  Sample layer names: {layer_names}{'...' if len(activation_dict) > 3 else ''}")

def main():
    # ============================================================================
    # AUTO-DISCOVER ACTIVATION FILES
    # ============================================================================
    
    print("\n" + "="*80)
    print("AUTO-DISCOVERING ACTIVATION FILES")
    print("="*80)
    
    # Base path for results
    activations_base_dir = os.path.join(RESULTS_DIR, "activations")
    
    # Model to analyze (you can make this a parameter later)
    target_model = "meta-llama_Llama-3.2-3B"
    model_dir = os.path.join(activations_base_dir, target_model)
    
    if not os.path.exists(model_dir):
        print(f"❌ Model directory not found: {model_dir}")
        print(f"   Available models:")
        if os.path.exists(activations_base_dir):
            for model_name in os.listdir(activations_base_dir):
                model_path = os.path.join(activations_base_dir, model_name)
                if os.path.isdir(model_path):
                    print(f"   - {model_name}")
        return
    
    print(f"✓ Found model directory: {target_model}")
    print(f"  Path: {model_dir}\n")
    
    # Discover all dataset directories
    dataset_dirs = []
    for dataset_name in os.listdir(model_dir):
        dataset_path = os.path.join(model_dir, dataset_name)
        if os.path.isdir(dataset_path):
            activation_file = os.path.join(dataset_path, "activations.pkl")
            if os.path.exists(activation_file):
                dataset_dirs.append((dataset_name, activation_file))
    
    if not dataset_dirs:
        print(f"❌ No activation files found in {model_dir}")
        return
    
    print(f"✓ Found {len(dataset_dirs)} dataset(s) with activations:")
    for dataset_name, activation_file in dataset_dirs:
        print(f"  - {dataset_name}")
    print()
    
    # ============================================================================
    # LOAD ALL DATASETS
    # ============================================================================
    
    print("="*80)
    print("LOADING DATASETS")
    print("="*80)
    
    loaded_datasets = []
    
    for dataset_name, activation_file in dataset_dirs:
        print(f"Loading: {dataset_name}...", end=" ")
        try:
            data = data_loader(activation_file)
            loaded_datasets.append((dataset_name, data))
            print("✓")
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    if not loaded_datasets:
        print("\n❌ No datasets loaded successfully!")
        return
    
    print(f"\n✓ Successfully loaded {len(loaded_datasets)} dataset(s)\n")
    
    # ============================================================================
    # OPTIONAL: STRUCTURE OVERVIEW
    # ============================================================================
    
    if loaded_datasets:
        print("="*80)
        print("DATASET STRUCTURE OVERVIEW (First Dataset)")
        print("="*80)
        #structure_explainer(loaded_datasets[0][1])
    
    # ============================================================================
    # ANALYZE MLP IMPACT
    # ============================================================================
    
    print("\n" + "="*80)
    print("STARTING MLP IMPACT ANALYSIS")
    print("="*80)
    print(f"Analyzing {len(loaded_datasets)} dataset(s):\n")
    
    for dataset_name, _ in loaded_datasets:
        print(f"  • {dataset_name}")
    print()
    
    # Run MLP impact analysis with model name
    compareMLPImpact(datasets=loaded_datasets, model_name=target_model)
    
    # ============================================================================
    # OPTIONAL: ADDITIONAL ANALYSES
    # ============================================================================
    
    # Uncomment below to run other analyses:
    
    # # Compute global means across all datasets
    # compute_global_means(loaded_datasets, embed="pre_ln2_activations")
    # print(f"\n✓ Global means computed for {len(global_means)} layers")
    
    # # Compare datasets in embedding space
    # print("\n" + "="*80)
    # print("EMBEDDING SPACE COMPARISONS")
    # print("="*80)
    # compareInEmbedSpaceSummary(datasets=loaded_datasets, embed="pre_ln1_activations")
    
    # # Domain consistency analysis (if you have paired datasets)
    # if len(loaded_datasets) >= 2:
    #     print("\n" + "="*80)
    #     print("DOMAIN CONSISTENCY ANALYSIS")
    #     print("="*80)
    #     compareDomainConsistency(
    #         loaded_datasets[0][1], 
    #         loaded_datasets[1][1], 
    #         embed="pre_ln2_activations"
    #     )

def value_check():

if __name__ == '__main__':
    #main()
    value_check()