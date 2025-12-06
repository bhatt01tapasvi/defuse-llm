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

def analyze_representation_structure(data, name, layer='block_11'):
    """
    Understand what's happening in your representations
    """
    mlp_acts = data['pre_ln2_activations'][layer][-1]  # (num_tokens, embed_dim)
    
    print(f"\n{'='*60}")
    print(f"Representation Analysis: {name} - {layer}")
    print(f"{'='*60}")
    
    # 1. Dimensionality analysis
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(mlp_acts)
    
    # How many dimensions capture 90%, 95%, 99% of variance?
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    dims_90 = np.argmax(cumsum >= 0.90) + 1
    dims_95 = np.argmax(cumsum >= 0.95) + 1
    dims_99 = np.argmax(cumsum >= 0.99) + 1
    
    print(f"\nEffective Dimensionality:")
    print(f"  90% variance captured by: {dims_90} dims (out of {mlp_acts.shape[1]})")
    print(f"  95% variance captured by: {dims_95} dims")
    print(f"  99% variance captured by: {dims_99} dims")
    print(f"  → Effective dim ratio: {dims_90/mlp_acts.shape[1]:.2%}")
    
    # 2. Sparsity analysis
    abs_acts = np.abs(mlp_acts)
    threshold_99 = np.percentile(abs_acts, 99)
    threshold_95 = np.percentile(abs_acts, 95)
    threshold_90 = np.percentile(abs_acts, 90)
    
    active_99 = (abs_acts > threshold_99).sum() / abs_acts.size
    active_95 = (abs_acts > threshold_95).sum() / abs_acts.size
    active_90 = (abs_acts > threshold_90).sum() / abs_acts.size
    
    print(f"\nActivation Sparsity:")
    print(f"  Top 1% threshold: {threshold_99:.4f}, Active: {active_99:.2%}")
    print(f"  Top 5% threshold: {threshold_95:.4f}, Active: {active_95:.2%}")
    print(f"  Top 10% threshold: {threshold_90:.4f}, Active: {active_90:.2%}")
    
    # 3. Last token analysis
    last_token = mlp_acts[-1]
    top_10_dims = np.argsort(np.abs(last_token))[-10:]
    
    print(f"\nLast Token Top-10 Dimensions:")
    print(f"  Indices: {top_10_dims}")
    print(f"  Values: {last_token[top_10_dims]}")
    print(f"  Magnitude: {np.linalg.norm(last_token):.4f}")
    
    return {
        'pca': pca,
        'dims_90': dims_90,
        'top_dims_last_token': top_10_dims,
        'last_token': last_token
    }

def analyze_mlp_residual_impact(pre_mlp_acts: Dict[str, list], post_mlp_acts: Dict[str, list]):
    """
    Compare residual stream before MLP vs after adding MLP output
    
    Residual connection: residual_after = residual_before + MLP(residual_before)
    
    Where:
    - residual_before = pre_mlp_acts (input to MLP, also passed through residual)
    - MLP_output = post_mlp_acts (what MLP computed)
    - residual_after = pre_mlp_acts + post_mlp_acts (final residual after adding MLP)
    
    We compare residual_before vs residual_after to see MLP's impact
    """
    
    results = []
    
    for layer_name in pre_mlp_acts.keys():
        # Single prefill pass - take the last (and only) forward pass
        residual_before = pre_mlp_acts[layer_name][-1]  # Shape: (num_tokens, embed_dim)
        mlp_output = post_mlp_acts[layer_name][-1]  # Shape: (num_tokens, embed_dim)
        
        # Construct the residual stream after adding MLP output
        residual_after = residual_before + mlp_output
        
        num_tokens = residual_before.shape[0]
        
        # Token-wise analysis
        token_similarities = []
        token_distances = []
        token_relative_changes = []
        token_mlp_norms = []
        token_before_norms = []
        token_after_norms = []
        
        for token_idx in range(num_tokens):
            before = residual_before[token_idx]
            after = residual_after[token_idx]
            mlp_delta = mlp_output[token_idx]
            
            # Metrics
            cos_sim = cosine_similarity(before, after)
            eucl_dist = euclidean_distance(before, after)
            
            before_norm = np.linalg.norm(before)
            after_norm = np.linalg.norm(after)
            mlp_norm = np.linalg.norm(mlp_delta)
            
            relative_change = mlp_norm / (before_norm + 1e-8)
            
            token_similarities.append(cos_sim)
            token_distances.append(eucl_dist)
            token_relative_changes.append(relative_change)
            token_mlp_norms.append(mlp_norm)
            token_before_norms.append(before_norm)
            token_after_norms.append(after_norm)
        
        # Aggregate for this layer
        layer_result = {
            'layer': layer_name,
            'avg_cosine_sim': np.mean(token_similarities),
            'min_cosine_sim': np.min(token_similarities),
            'max_cosine_sim': np.max(token_similarities),
            'avg_euclidean': np.mean(token_distances),
            'avg_relative_change': np.mean(token_relative_changes),
            'max_relative_change': np.max(token_relative_changes),
            'avg_mlp_norm': np.mean(token_mlp_norms),
            'avg_before_norm': np.mean(token_before_norms),
            'avg_after_norm': np.mean(token_after_norms),
        }
        
        results.append(layer_result)
    
    return results

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

def compareMLPImpact(datasets: list[Tuple[str, Any]]):
    """
    Analyze MLP impact for multiple datasets
    """
    
    print("\n" + "="*80)
    print("MLP RESIDUAL STREAM IMPACT ANALYSIS")
    print("Comparing: Residual_Before vs Residual_After = Residual_Before + MLP_Output")
    print("="*80)
    
    all_results = {}
    
    for dataset_name, data in datasets:
    #     # Check if the residual connection math is consistent
    #     # Compare layer by layer, token by token
    #     is_consistent = True
        
    #     for layer_name in data['pre_ln2_activations'].keys():
    #         print(len(data['pre_ln2_activations'][layer_name]),len(data['pre_ln2_activations'][layer_name][0][0]))
    #         pre_ln2 = data['pre_ln2_activations'][layer_name][-1]  # (num_tokens, embed_dim)
    #         post_mlp2 = data['post_mlp2_activations'][layer_name][-1]  # (num_tokens, embed_dim)
    #         post_layer = data['post_layer_activations'][layer_name][-1]  # (num_tokens, embed_dim)
            
    #         # Check if: post_layer = pre_ln2 + post_mlp2
    #         expected = pre_ln2 + post_mlp2
            
    #         if not np.allclose(expected, post_layer, rtol=1e-5, atol=1e-6):
    #             print(f"⚠️  Data inconsistency in dataset: {dataset_name}, layer: {layer_name}")
    #             print(f"    Expected shape: {expected.shape}, Got: {post_layer.shape}")
    #             print(f"    Max difference: {np.max(np.abs(expected - post_layer)):.6f}")
    #             is_consistent = False
    #             break
        
    #     if not is_consistent:
    #         print(f"❌ Skipping analysis for {dataset_name} due to inconsistency.\n")
    #         continue
        
    #    print(f"✓ Data consistency verified for {dataset_name}")
        
        # Analyze this dataset
        results = analyze_mlp_residual_impact(
            data['pre_ln2_activations'],  # Residual before MLP
            data['post_mlp2_activations']  # MLP output (to be added to residual)
        )
        
        # Save results
        save_path = os.path.join(MLP_IMPACT_DIR, f"{dataset_name}_mlp_impact.csv")
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
    # activation_imc = "results/prompts/long/imc/activations.pkl"
    # data_imc = data_loader(activation_imc)

    # activation_astro = "results/prompts/long/astro/activations.pkl"
    # data_astro = data_loader(activation_astro)

    # activation_imc2 = "results/prompts/long/imc2/activations.pkl"
    # data_imc2 = data_loader(activation_imc2)

    # activation_imc3 = "results/prompts/long/imc3/activations.pkl"
    # data_imc3 = data_loader(activation_imc3)

    # activation_pizzas = "results/prompts/long/pizzas/activations.pkl"
    # data_pizzas = data_loader(activation_pizzas)

    # activation_actress = "results/prompts/long/actress/activations.pkl"
    # data_actress = data_loader(activation_actress)

    # activation_hindi = "results/prompts/long/hindi/activations.pkl"
    # data_hindi = data_loader(activation_hindi)

    # activation_maths = "results/prompts/long/maths/activations.pkl"
    # data_maths = data_loader(activation_maths)

    # activation_imc_word = "results/prompts/short/imc_word/activations.pkl"
    # data_imc_word = data_loader(activation_imc_word)

    # activation_imc2_word = "results/prompts/short/imc2_word/activations.pkl"
    # data_imc2_word = data_loader(activation_imc2_word)

    # activation_pizzas_word = "results/prompts/short/pizzas_word/activations.pkl"
    # data_pizzas_word = data_loader(activation_pizzas_word)

    # activation_actress_word = "results/prompts/short/actress_word/activations.pkl"
    # data_actress_word = data_loader(activation_actress_word)

    # activation_astro_word = "results/prompts/short/astro_word/activations.pkl"
    # data_astro_word = data_loader(activation_astro_word)

    # activation_imc2_para = "results/prompts/para/imc2/activations.pkl"
    # data_imc2_para = data_loader(activation_imc2_para)

    # activation_imc_para = "results/prompts/para/imc/activations.pkl"
    # data_imc_para = data_loader(activation_imc_para)
    
    # activation_pizzas_para = "results/prompts/para/pizzas/activations.pkl"
    # data_pizzas_para = data_loader(activation_pizzas_para)

    # activation_actress_para = "results/prompts/para/actress/activations.pkl"
    # data_actress_para = data_loader(activation_actress_para)

    # activation_gpt_ask1 = "results/prompts/gpt/ask1/activations.pkl"
    # data_gpt_ask1 = data_loader(activation_gpt_ask1)

    # activation_gpt_reply1 = "results/prompts/gpt/reply1/activations.pkl"
    # data_gpt_reply1 = data_loader(activation_gpt_reply1)

    # all_datasets = [
    #     ("IMC", data_imc), ("Astro", data_astro), ("IMC2", data_imc2), ("IMC3", data_imc3),
    #     ("Pizzas", data_pizzas), ("Actress", data_actress), ("Hindi", data_hindi), ("Maths", data_maths),
    #     ("IMC_word", data_imc_word), ("IMC2_word", data_imc2_word), ("Pizzas_word", data_pizzas_word),
    #     ("Actress_word", data_actress_word), ("Astro_word", data_astro_word),
    #     ("IMC2_para", data_imc2_para), ("IMC_para", data_imc_para), 
    #     ("Pizzas_para", data_pizzas_para), ("Actress_para", data_actress_para),
    #     ("GPT_ask1", data_gpt_ask1), ("GPT_reply1", data_gpt_reply1)
    # ]

    # normal_datasets = [
    #     ("IMC", data_imc), ("Astro", data_astro), ("IMC2", data_imc2), ("IMC3", data_imc3),
    #     ("Pizzas", data_pizzas), ("Actress", data_actress), ("Hindi", data_hindi), ("Maths", data_maths)
    # ]

    # word_datasets = [
    #     ("IMC_word", data_imc_word), ("IMC2_word", data_imc2_word), ("Pizzas_word", data_pizzas_word),
    #     ("Actress_word", data_actress_word), ("Astro_word", data_astro_word)
    # ]

    # para_datasets = [
    #     ("IMC2_para", data_imc2_para), ("IMC_para", data_imc_para),
    #     ("Pizzas_para", data_pizzas_para), ("Actress_para", data_actress_para)
    # ]

    #compute_global_means(all_datasets, embed="pre_ln2_activations")
    #print(f"\nGlobal means computed for {len(global_means)} layers")

    # Quick overview of the structure of the datasets
    #structure_explainer(data_imc)

    # print("\n" + "="*80)
    # print("EMBEDDING SPACE COMPARISONS")
    # print("="*80)

    #compareInEmbedSpaceSummary(datasets=normal_datasets, embed = "pre_ln1_activations")

    #compareDomainConsistency(data_gpt_ask1, data_gpt_reply1, embed="pre_ln2_activations")

    # Run for both
    # result_imc = analyze_representation_structure(data_imc_word, "In-memory computing")
    # result_pizza = analyze_representation_structure(data_pizzas_word, "Neapolitan pizza")
    # result_imc2 = analyze_representation_structure(data_imc2_word, "In-memory computing 2")
    # result_actress = analyze_representation_structure(data_actress_word, "Actress")

    activation_abstract_algebra = "/users/grad/abhishektyagi/wanda/wanda/results/activations/meta-llama_Llama-3.2-3B/custom_abstract_algebra_corpus/activations.pkl"
    data_abstract_algebra = data_loader(activation_abstract_algebra)

    new_datasets = [
        ("abstract_algebra", data_abstract_algebra)
    ]

    compareMLPImpact(datasets=new_datasets)

if __name__ == '__main__':
    main()