import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
from util import data_loader
from scipy.spatial.distance import cosine
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt

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
    
def flatten_activation_structure(activations_dict: Dict[str, list]) -> Dict[str, list]:
    """
    Flatten activation structure so each element is a single token (1, embed_dim).
    
    Input structure:
    - activations_dict[layer][0] has shape (num_prefill_tokens, embed_dim)
    - activations_dict[layer][1:] each have shape (1, embed_dim)
    
    Output structure:
    - activations_dict[layer][i] has shape (1, embed_dim) for all i
    
    Args:
        activations_dict: Dictionary with layer names as keys and lists of activations
        
    Returns:
        Flattened dictionary with same structure but each element is single token
    """
    flattened = {}
    
    for layer_name, activations_list in activations_dict.items():
        flattened_list = []
        
        for idx, activation in enumerate(activations_list):
            if idx == 0 and activation.shape[0] > 1:
                # This is the prefill phase with multiple tokens
                # Split into individual tokens, each with shape (1, embed_dim)
                for token_idx in range(activation.shape[0]):
                    single_token = activation[token_idx:token_idx+1]  # Keep 2D shape
                    flattened_list.append(single_token)
            else:
                # Already a single token, just append
                flattened_list.append(activation)
        
        flattened[layer_name] = flattened_list
    
    return flattened

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

def compare_t_t_Consistency(model, data_a: Dict[str, Any], data_b: Dict[str, Any], embed: str, topic: str = "default", layer_num: int = -1):
    """
    Compare two sequences token-by-token using cosine similarity.
    
    After flattening, each element in the activation list is a single token with shape (1, embed_dim).
    This function compares corresponding tokens between data_a and data_b.
    
    Args:
        data_a: First dataset (e.g., unpruned model)
        data_b: Second dataset (e.g., pruned model)
        embed: Key for the activation type to compare (e.g., 'pre_ln2_activations')
        topic: Topic name for organizing results (default: 'default')
        layer_num: Layer index to analyze (default: -1 for last layer)
    """
    
    print("\n" + "="*80)
    print("TOKEN-BY-TOKEN DOMAIN CONSISTENCY ANALYSIS")
    print("="*80)
    
    # Get the specified layer
    all_layers = list(data_a[embed].keys())
    layer = all_layers[layer_num]
    
    print(f"Analyzing layer: {layer} (index: {layer_num})")
    
    # Get token lists (after flattening, each element is a single token with shape (1, embed_dim))
    tokens_a = data_a[embed][layer]
    tokens_b = data_b[embed][layer]
    
    num_tokens_a = len(tokens_a)
    num_tokens_b = len(tokens_b)
    
    print(f"Number of tokens in data_a: {num_tokens_a}")
    print(f"Number of tokens in data_b: {num_tokens_b}")
    
    # Use the minimum number of tokens for comparison
    num_tokens = min(num_tokens_a, num_tokens_b)
    print(f"Comparing first {num_tokens} tokens\n")
    
    # Store token-by-token comparisons
    comparison_results = []
    
    for token_idx in range(num_tokens):
        # Get individual token embeddings (shape: (1, embed_dim))
        token_a = tokens_a[token_idx].squeeze(0)  # Shape: (embed_dim,)
        token_b = tokens_b[token_idx].squeeze(0)  # Shape: (embed_dim,)
        
        # Compute cosine similarity
        cos_sim = cosine_similarity(token_a, token_b)
        
        # Compute Euclidean distance
        eucl_dist = euclidean_distance(token_a, token_b)
        
        # Compute magnitude ratio
        mag_a = np.linalg.norm(token_a)
        mag_b = np.linalg.norm(token_b)
        mag_ratio = mag_a / (mag_b + 1e-8)
        
        # Jaccard similarity on top activated dimensions
        top_100_a = set(np.argsort(np.abs(token_a))[-100:])
        top_100_b = set(np.argsort(np.abs(token_b))[-100:])
        jaccard_sim = len(top_100_a & top_100_b) / len(top_100_a | top_100_b)
        
        comparison_results.append({
            'Token Position': token_idx + 1,
            'Cosine Similarity': cos_sim,
            'Euclidean Distance': eucl_dist,
            'Magnitude Ratio (A/B)': mag_ratio,
            'Jaccard Similarity': jaccard_sim,
            'Magnitude A': mag_a,
            'Magnitude B': mag_b
        })
    
    # Create DataFrame
    df = pd.DataFrame(comparison_results)
    
    # Print summary statistics
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Average Cosine Similarity: {df['Cosine Similarity'].mean():.4f}")
    print(f"Min Cosine Similarity: {df['Cosine Similarity'].min():.4f} (Token {df['Cosine Similarity'].idxmin() + 1})")
    print(f"Max Cosine Similarity: {df['Cosine Similarity'].max():.4f} (Token {df['Cosine Similarity'].idxmax() + 1})")
    print(f"Std Dev Cosine Similarity: {df['Cosine Similarity'].std():.4f}")
    print(f"\nAverage Euclidean Distance: {df['Euclidean Distance'].mean():.2f}")
    print(f"Average Jaccard Similarity: {df['Jaccard Similarity'].mean():.4f}")
    print(f"Average Magnitude Ratio: {df['Magnitude Ratio (A/B)'].mean():.4f}")
    
    # Save to CSV with hierarchical folder structure
    topic_dir = os.path.join(DOMAIN_CONSISTENCY_DIR, model, topic)
    layer_dir = os.path.join(topic_dir, layer)
    embed_dir = os.path.join(layer_dir, embed)
    os.makedirs(embed_dir, exist_ok=True)
    
    save_path = os.path.join(embed_dir, "token_by_token_comparison.csv")
    df.to_csv(save_path, index=False)
    print(f"\n✓ Detailed results saved to: {save_path}")
    print("\n")
    
    return df

def compare_c_t_Consistency(model, data: Dict[str, Any], embed: str, data_name: str, topic: str = "default", masking_point: int = 120, layer_num: int = -1):
    """
    Compare each token against a center point (mean of tokens up to masking point).
    
    This function computes a representative "center" vector as the mean of all tokens
    up to the masking point, then compares each token against this center using
    cosine similarity and other metrics.
    
    Args:
        data: Dataset to analyze
        embed: Key for the activation type to compare (e.g., 'pre_ln2_activations')
        data_name: Name of the dataset (for output file naming)
        topic: Topic name for organizing results (default: 'default')
        masking_point: Token position up to which to compute the center (default: 120)
        layer_num: Layer index to analyze (default: -1 for last layer)
    """
    
    print("\n" + "="*80)
    print(f"CENTER-TO-TOKEN CONSISTENCY ANALYSIS: {data_name}")
    print("="*80)
    
    # Get the specified layer
    all_layers = list(data[embed].keys())
    layer = all_layers[layer_num]
    
    print(f"Analyzing layer: {layer} (index: {layer_num})")
    
    # Get token list (after flattening, each element is a single token with shape (1, embed_dim))
    tokens = data[embed][layer]
    num_tokens = len(tokens)
    
    print(f"Number of tokens: {num_tokens}")
    print(f"Masking point: {masking_point}")
    
    # Compute center vector as mean of tokens up to masking point
    if num_tokens < masking_point:
        print(f"⚠️  Warning: Only {num_tokens} tokens available, using all for center computation")
        masking_point = num_tokens
    
    # Collect all tokens up to masking point
    center_tokens = []
    for token_idx in range(masking_point):
        token = tokens[token_idx].squeeze(0)  # Shape: (embed_dim,)
        center_tokens.append(token)
    
    # Compute center as mean
    center_vector = np.mean(center_tokens, axis=0)  # Shape: (embed_dim,)
    
    print(f"Center vector computed from first {masking_point} tokens")
    
    # Store center-to-token comparisons
    comparison_results = []
    
    for token_idx in range(num_tokens):
        # Get individual token embedding
        token = tokens[token_idx].squeeze(0)  # Shape: (embed_dim,)
        
        # Compute cosine similarity with center
        cos_sim = cosine_similarity(token, center_vector)
        
        # Compute Euclidean distance from center
        eucl_dist = euclidean_distance(token, center_vector)
        
        # Compute magnitude ratio
        token_mag = np.linalg.norm(token)
        center_mag = np.linalg.norm(center_vector)
        mag_ratio = token_mag / (center_mag + 1e-8)
        
        # Jaccard similarity on top activated dimensions
        top_100_token = set(np.argsort(np.abs(token))[-100:])
        top_100_center = set(np.argsort(np.abs(center_vector))[-100:])
        jaccard_sim = len(top_100_token & top_100_center) / len(top_100_token | top_100_center)
        
        # Mark if this token was used in center computation
        used_in_center = "Yes" if token_idx < masking_point else "No"
        
        comparison_results.append({
            'Token Position': token_idx + 1,
            'Used in Center': used_in_center,
            'Cosine Similarity': cos_sim,
            'Euclidean Distance': eucl_dist,
            'Magnitude Ratio (Token/Center)': mag_ratio,
            'Jaccard Similarity': jaccard_sim,
            'Token Magnitude': token_mag,
            'Center Magnitude': center_mag
        })
    
    # Create DataFrame
    df = pd.DataFrame(comparison_results)
    
    # Print summary statistics
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Average Cosine Similarity (all tokens): {df['Cosine Similarity'].mean():.4f}")
    print(f"Min Cosine Similarity: {df['Cosine Similarity'].min():.4f} (Token {df['Cosine Similarity'].idxmin() + 1})")
    print(f"Max Cosine Similarity: {df['Cosine Similarity'].max():.4f} (Token {df['Cosine Similarity'].idxmax() + 1})")
    print(f"Std Dev Cosine Similarity: {df['Cosine Similarity'].std():.4f}")
    
    # Statistics for tokens used in center vs. not used
    center_tokens_df = df[df['Used in Center'] == 'Yes']
    non_center_tokens_df = df[df['Used in Center'] == 'No']
    
    if len(center_tokens_df) > 0:
        print(f"\nTokens used in center (first {masking_point}):")
        print(f"  Average Cosine Similarity: {center_tokens_df['Cosine Similarity'].mean():.4f}")
    
    if len(non_center_tokens_df) > 0:
        print(f"\nTokens NOT used in center (after {masking_point}):")
        print(f"  Average Cosine Similarity: {non_center_tokens_df['Cosine Similarity'].mean():.4f}")
    
    print(f"\nAverage Euclidean Distance: {df['Euclidean Distance'].mean():.2f}")
    print(f"Average Jaccard Similarity: {df['Jaccard Similarity'].mean():.4f}")
    print(f"Average Magnitude Ratio: {df['Magnitude Ratio (Token/Center)'].mean():.4f}")
    
    # Save to CSV with hierarchical folder structure
    topic_dir = os.path.join(DOMAIN_CONSISTENCY_DIR, model, topic)
    layer_dir = os.path.join(topic_dir, layer)
    embed_dir = os.path.join(layer_dir, embed)
    os.makedirs(embed_dir, exist_ok=True)
    
    save_path = os.path.join(embed_dir, f"center_by_token_{data_name}.csv")
    df.to_csv(save_path, index=False)
    print(f"\n✓ Detailed results saved to: {save_path}")
    print("\n")
    
    return df

def compare_c_w_Consistency(model, data: Dict[str, Any], embed: str, data_name: str, topic: str = "default", masking_point: int = 120, window_size: int = 10, layer_num: int = -1):
    """
    Compare window centers against masked region center.
    
    This function groups tokens into windows of specified size and computes a center
    for each window. Each window center is then compared against the global center
    (computed from the first masking_point tokens) to track how similarity changes
    as the topic shifts.
    
    Args:
        model: Model name (for file organization)
        data: Dataset to analyze
        embed: Key for the activation type to compare (e.g., 'pre_ln2_activations')
        data_name: Name of the dataset (for output file naming)
        topic: Topic name for organizing results (default: 'default')
        masking_point: Token position up to which to compute the global center (default: 120)
        window_size: Number of tokens in each window (default: 10)
        layer_num: Layer index to analyze (default: -1 for last layer)
    """
    
    print("\n" + "="*80)
    print(f"CENTER-TO-WINDOW CONSISTENCY ANALYSIS: {data_name}")
    print("="*80)
    
    # Get the specified layer
    all_layers = list(data[embed].keys())
    layer = all_layers[layer_num]
    
    print(f"Analyzing layer: {layer} (index: {layer_num})")
    
    # Get token list (after flattening, each element is a single token with shape (1, embed_dim))
    tokens = data[embed][layer]
    num_tokens = len(tokens)
    
    print(f"Number of tokens: {num_tokens}")
    print(f"Masking point: {masking_point}")
    print(f"Window size: {window_size}")
    
    # Compute global center vector as mean of tokens up to masking point
    if num_tokens < masking_point:
        print(f"⚠️  Warning: Only {num_tokens} tokens available, using all for global center computation")
        masking_point = num_tokens
    
    # Collect all tokens up to masking point for global center
    global_center_tokens = []
    for token_idx in range(masking_point):
        token = tokens[token_idx].squeeze(0)  # Shape: (embed_dim,)
        global_center_tokens.append(token)
    
    # Compute global center as mean
    global_center = np.mean(global_center_tokens, axis=0)  # Shape: (embed_dim,)
    
    print(f"Global center computed from first {masking_point} tokens")
    
    # Group all tokens into windows
    num_windows = num_tokens // window_size
    if num_tokens % window_size != 0:
        num_windows += 1  # Include partial window at the end
    
    print(f"Total windows: {num_windows}")
    
    # Store window-to-global-center comparisons
    comparison_results = []
    
    for window_idx in range(num_windows):
        window_start = window_idx * window_size
        window_end = min((window_idx + 1) * window_size, num_tokens)
        
        # Collect tokens in this window
        window_tokens = []
        for token_idx in range(window_start, window_end):
            token = tokens[token_idx].squeeze(0)  # Shape: (embed_dim,)
            window_tokens.append(token)
        
        # Compute window center
        window_center = np.mean(window_tokens, axis=0)  # Shape: (embed_dim,)
        
        # Compute metrics between window center and global center
        cos_sim = cosine_similarity(window_center, global_center)
        eucl_dist = euclidean_distance(window_center, global_center)
        
        # Compute magnitude ratio
        window_mag = np.linalg.norm(window_center)
        global_mag = np.linalg.norm(global_center)
        mag_ratio = window_mag / (global_mag + 1e-8)
        
        # Jaccard similarity on top activated dimensions
        top_100_window = set(np.argsort(np.abs(window_center))[-100:])
        top_100_global = set(np.argsort(np.abs(global_center))[-100:])
        jaccard_sim = len(top_100_window & top_100_global) / len(top_100_window | top_100_global)
        
        # Check if window overlaps with masked region
        overlaps_mask = window_start < masking_point
        
        comparison_results.append({
            'Window Index': window_idx + 1,
            'Token Range': f"{window_start + 1}-{window_end}",
            'Window Size': len(window_tokens),
            'Overlaps Mask': "Yes" if overlaps_mask else "No",
            'Cosine Similarity': cos_sim,
            'Euclidean Distance': eucl_dist,
            'Magnitude Ratio (Window/Global)': mag_ratio,
            'Jaccard Similarity': jaccard_sim,
            'Window Magnitude': window_mag,
            'Global Magnitude': global_mag
        })
    
    # Create DataFrame
    df = pd.DataFrame(comparison_results)
    
    # Print summary statistics
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Average Cosine Similarity (all windows): {df['Cosine Similarity'].mean():.4f}")
    print(f"Min Cosine Similarity: {df['Cosine Similarity'].min():.4f} (Window {df['Cosine Similarity'].idxmin() + 1})")
    print(f"Max Cosine Similarity: {df['Cosine Similarity'].max():.4f} (Window {df['Cosine Similarity'].idxmax() + 1})")
    print(f"Std Dev Cosine Similarity: {df['Cosine Similarity'].std():.4f}")
    
    # Statistics for windows that overlap vs don't overlap with mask
    overlap_windows_df = df[df['Overlaps Mask'] == 'Yes']
    non_overlap_windows_df = df[df['Overlaps Mask'] == 'No']
    
    if len(overlap_windows_df) > 0:
        print(f"\nWindows overlapping with mask (before token {masking_point}):")
        print(f"  Count: {len(overlap_windows_df)}")
        print(f"  Average Cosine Similarity: {overlap_windows_df['Cosine Similarity'].mean():.4f}")
    
    if len(non_overlap_windows_df) > 0:
        print(f"\nWindows NOT overlapping with mask (after token {masking_point}):")
        print(f"  Count: {len(non_overlap_windows_df)}")
        print(f"  Average Cosine Similarity: {non_overlap_windows_df['Cosine Similarity'].mean():.4f}")
        print(f"  Similarity Range: [{non_overlap_windows_df['Cosine Similarity'].min():.4f}, {non_overlap_windows_df['Cosine Similarity'].max():.4f}]")
    
    print(f"\nAverage Euclidean Distance: {df['Euclidean Distance'].mean():.2f}")
    print(f"Average Jaccard Similarity: {df['Jaccard Similarity'].mean():.4f}")
    print(f"Average Magnitude Ratio: {df['Magnitude Ratio (Window/Global)'].mean():.4f}")
    
    # Save to CSV with hierarchical folder structure
    topic_dir = os.path.join(DOMAIN_CONSISTENCY_DIR, model, topic)
    layer_dir = os.path.join(topic_dir, layer)
    embed_dir = os.path.join(layer_dir, embed)
    os.makedirs(embed_dir, exist_ok=True)
    
    save_path = os.path.join(embed_dir, f"center_by_window_{data_name}.csv")
    df.to_csv(save_path, index=False)
    print(f"\n✓ Detailed results saved to: {save_path}")
    print("\n")
    
    return df

def compare_attn_oproj_tokenwise(model, data: Dict[str, Any], embed: str, data_name: str, topic: str = "default", 
                                   num_reference_tokens: int = 120, layer_num: int = -1):
    """
    Compare each token's attention output projection against the average of reference tokens.
    
    Computes a reference center vector as the mean of attention output projections from
    the first `num_reference_tokens` tokens, then compares each token against this center
    using cosine similarity, euclidean distance, magnitude ratio, and Jaccard similarity.
    
    Args:
        model: Model name (for file organization)
        data: Dataset to analyze
        embed: Key for the activation type (e.g., 'post_attn_activations')
        data_name: Name of the dataset (for output file naming)
        topic: Topic name for organizing results (default: 'default')
        num_reference_tokens: Number of initial tokens to compute reference center (default: 120)
        layer_num: Layer index to analyze (default: -1 for last layer)
    
    Returns:
        DataFrame with token-wise comparison results
    """
    
    print("\n" + "="*80)
    print(f"ATTENTION OUTPUT PROJECTION TOKEN-WISE ANALYSIS: {data_name}")
    print("="*80)
    
    # Get the specified layer
    all_layers = list(data[embed].keys())
    layer = all_layers[layer_num]
    
    print(f"Analyzing layer: {layer} (index: {layer_num})")
    
    # Get token list (after flattening, each element is a single token with shape (1, embed_dim))
    tokens = data[embed][layer]
    num_tokens = len(tokens)
    
    print(f"Number of tokens: {num_tokens}")
    print(f"Number of reference tokens for center: {num_reference_tokens}")
    
    # Compute reference center vector as mean of first num_reference_tokens
    if num_tokens < num_reference_tokens:
        print(f"⚠️  Warning: Only {num_tokens} tokens available, using all for reference center computation")
        num_reference_tokens = num_tokens
    
    # Collect reference tokens
    reference_tokens = []
    for token_idx in range(num_reference_tokens):
        token = tokens[token_idx].squeeze(0)  # Shape: (embed_dim,)
        reference_tokens.append(token)
    
    # Compute reference center as mean
    reference_center = np.mean(reference_tokens, axis=0)  # Shape: (embed_dim,)
    
    print(f"Reference center computed from first {num_reference_tokens} tokens")
    
    # Store token-wise comparisons
    comparison_results = []
    
    for token_idx in range(num_tokens):
        # Get individual token embedding
        token = tokens[token_idx].squeeze(0)  # Shape: (embed_dim,)
        
        # Compute cosine similarity with reference center
        cos_sim = cosine_similarity(token, reference_center)
        
        # Compute Euclidean distance from reference center
        eucl_dist = euclidean_distance(token, reference_center)
        
        # Compute magnitude ratio
        token_mag = np.linalg.norm(token)
        center_mag = np.linalg.norm(reference_center)
        mag_ratio = token_mag / (center_mag + 1e-8)
        
        # Jaccard similarity on top activated dimensions
        top_100_token = set(np.argsort(np.abs(token))[-100:])
        top_100_center = set(np.argsort(np.abs(reference_center))[-100:])
        jaccard_sim = len(top_100_token & top_100_center) / len(top_100_token | top_100_center)
        
        # Mark if this token was used in reference center computation
        used_in_reference = "Yes" if token_idx < num_reference_tokens else "No"
        
        comparison_results.append({
            'Token Position': token_idx + 1,
            'Used in Reference': used_in_reference,
            'Cosine Similarity': cos_sim,
            'Euclidean Distance': eucl_dist,
            'Magnitude Ratio (Token/Reference)': mag_ratio,
            'Jaccard Similarity': jaccard_sim,
            'Token Magnitude': token_mag,
            'Reference Magnitude': center_mag
        })
    
    # Create DataFrame
    df = pd.DataFrame(comparison_results)
    
    # Print summary statistics
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Average Cosine Similarity (all tokens): {df['Cosine Similarity'].mean():.4f}")
    print(f"Min Cosine Similarity: {df['Cosine Similarity'].min():.4f} (Token {df['Cosine Similarity'].idxmin() + 1})")
    print(f"Max Cosine Similarity: {df['Cosine Similarity'].max():.4f} (Token {df['Cosine Similarity'].idxmax() + 1})")
    print(f"Std Dev Cosine Similarity: {df['Cosine Similarity'].std():.4f}")
    
    # Statistics for tokens used in reference vs. not used
    reference_tokens_df = df[df['Used in Reference'] == 'Yes']
    non_reference_tokens_df = df[df['Used in Reference'] == 'No']
    
    if len(reference_tokens_df) > 0:
        print(f"\nTokens used in reference (first {num_reference_tokens}):")
        print(f"  Average Cosine Similarity: {reference_tokens_df['Cosine Similarity'].mean():.4f}")
    
    if len(non_reference_tokens_df) > 0:
        print(f"\nTokens NOT used in reference (after {num_reference_tokens}):")
        print(f"  Average Cosine Similarity: {non_reference_tokens_df['Cosine Similarity'].mean():.4f}")
    
    print(f"\nAverage Euclidean Distance: {df['Euclidean Distance'].mean():.2f}")
    print(f"Average Jaccard Similarity: {df['Jaccard Similarity'].mean():.4f}")
    print(f"Average Magnitude Ratio: {df['Magnitude Ratio (Token/Reference)'].mean():.4f}")

    cosine_sim_token = cosine_similarity(tokens[2].squeeze(0), tokens[21].squeeze(0))
    cosine_sim_token_2 = cosine_similarity(tokens[3].squeeze(0), tokens[21].squeeze(0))
    cosine_sim_token_3 = cosine_similarity(tokens[3].squeeze(0), tokens[2].squeeze(0))

    print(f"\nCosine Similarity between token 3 and token 22: {cosine_sim_token:.4f}")
    print(f"Cosine Similarity between token 4 and token 22: {cosine_sim_token_2:.4f}")
    print(f"Cosine Similarity between token 4 and token 3: {cosine_sim_token_3:.4f}")
    # Save to CSV with hierarchical folder structure
    topic_dir = os.path.join(DOMAIN_CONSISTENCY_DIR, model, topic)
    layer_dir = os.path.join(topic_dir, layer)
    embed_dir = os.path.join(layer_dir, embed)
    os.makedirs(embed_dir, exist_ok=True)
    
    save_path = os.path.join(embed_dir, f"attn_oproj_tokenwise_{data_name}.csv")
    df.to_csv(save_path, index=False)
    print(f"\n✓ Detailed results saved to: {save_path}")
    print("\n")
    
    return df

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

def analyze_weight_activation_distributions(data: dict, save_dir: str = None):
    """
    Analyze and visualize the distribution of weights vs activations.
    
    Structure:
    - mlp2_weights[layer]: list of 3072 arrays, each of shape (8192,)
      → These are the weight COLUMNS (one per output neuron)
    - pre_mlp2_activations[layer]: list with 1 array of shape (1, 4, 8192)
      → These are the neuron activations (batch, tokens, neurons)
    """
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("WEIGHT vs ACTIVATION DISTRIBUTION ANALYSIS")
    print("="*80)
    
    mlp2_weights = data['mlp2_weights']
    pre_mlp2_activations = data['pre_mlp2_activations']
    
    # Choose a few representative layers to analyze
    all_layers = list(mlp2_weights.keys())
    # Analyze first, middle, and last layers
    layers_to_analyze = [
        all_layers[0],           # First layer
        all_layers[len(all_layers)//2],  # Middle layer
        all_layers[-1]           # Last layer
    ]
    
    print(f"\nAnalyzing {len(layers_to_analyze)} representative layers:")
    for layer in layers_to_analyze:
        print(f"  • {layer}")
    print()
    
    summary_data = []
    
    for layer_idx, layer_name in enumerate(layers_to_analyze):
        print(f"\n{'='*60}")
        print(f"Layer: {layer_name}")
        print(f"{'='*60}")
        
        # ====================================================================
        # EXTRACT WEIGHTS
        # ====================================================================
        # mlp2_weights[layer] is a list of 3072 arrays, each shape (8192,)
        # This represents the weight matrix columns (input_dim=8192, output_dim=3072)
        weights_list = mlp2_weights[layer_name]
        
        # Stack to get full weight matrix: (output_neurons=3072, input_neurons=8192)
        weights_matrix = np.stack(weights_list, axis=0)  # Shape: (3072, 8192)
        
        # Flatten all weights for distribution analysis
        all_weights = weights_matrix.flatten()
        
        print(f"\nWeights:")
        print(f"  Shape: {weights_matrix.shape} (output_neurons, input_neurons)")
        print(f"  Total elements: {all_weights.size:,}")
        print(f"  Mean: {np.mean(all_weights):.6f}")
        print(f"  Std: {np.std(all_weights):.6f}")
        print(f"  Min: {np.min(all_weights):.6f}")
        print(f"  Max: {np.max(all_weights):.6f}")
        print(f"  Median: {np.median(all_weights):.6f}")
        print(f"  25th percentile: {np.percentile(all_weights, 25):.6f}")
        print(f"  75th percentile: {np.percentile(all_weights, 75):.6f}")
        
        # ====================================================================
        # EXTRACT ACTIVATIONS
        # ====================================================================
        # pre_mlp2_activations[layer] is a list with 1 element of shape (1, 4, 8192)
        # → (batch_size=1, num_tokens=4, num_neurons=8192)
        activations_list = pre_mlp2_activations[layer_name]
        activations_array = activations_list[0]  # Shape: (1, 4, 8192)
        
        # Remove batch dimension and flatten
        activations_array = activations_array.squeeze(0)  # Shape: (4, 8192)
        all_activations = activations_array.flatten()
        
        print(f"\nActivations:")
        print(f"  Shape: {activations_array.shape} (num_tokens, num_neurons)")
        print(f"  Total elements: {all_activations.size:,}")
        print(f"  Mean: {np.mean(all_activations):.6f}")
        print(f"  Std: {np.std(all_activations):.6f}")
        print(f"  Min: {np.min(all_activations):.6f}")
        print(f"  Max: {np.max(all_activations):.6f}")
        print(f"  Median: {np.median(all_activations):.6f}")
        print(f"  25th percentile: {np.percentile(all_activations, 25):.6f}")
        print(f"  75th percentile: {np.percentile(all_activations, 75):.6f}")
        
        # ====================================================================
        # COMPUTE SCALE RATIOS
        # ====================================================================
        weight_scale = np.std(all_weights)
        activation_scale = np.std(all_activations)
        scale_ratio = activation_scale / (weight_scale + 1e-12)
        
        weight_magnitude = np.mean(np.abs(all_weights))
        activation_magnitude = np.mean(np.abs(all_activations))
        magnitude_ratio = activation_magnitude / (weight_magnitude + 1e-12)
        
        print(f"\nScale Comparison:")
        print(f"  Weight std / Activation std: {1/scale_ratio:.4f} : 1")
        print(f"  Activation std / Weight std: {scale_ratio:.4f} : 1")
        print(f"  Mean |weight| / Mean |activation|: {1/magnitude_ratio:.4f} : 1")
        print(f"  Mean |activation| / Mean |weight|: {magnitude_ratio:.4f} : 1")
        
        # ====================================================================
        # SAMPLE VALUES FOR INSPECTION
        # ====================================================================
        print(f"\nSample Values (first 10):")
        print(f"  Weights:     {all_weights[:10]}")
        print(f"  Activations: {all_activations[:10]}")
        
        # ====================================================================
        # STORE SUMMARY DATA
        # ====================================================================
        summary_data.append({
            'Layer': layer_name,
            'Weight Mean': np.mean(all_weights),
            'Weight Std': np.std(all_weights),
            'Weight Min': np.min(all_weights),
            'Weight Max': np.max(all_weights),
            'Activation Mean': np.mean(all_activations),
            'Activation Std': np.std(all_activations),
            'Activation Min': np.min(all_activations),
            'Activation Max': np.max(all_activations),
            'Scale Ratio (Act/Weight)': scale_ratio,
            'Magnitude Ratio (Act/Weight)': magnitude_ratio
        })
        
        # ====================================================================
        # VISUALIZATION
        # ====================================================================
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{layer_name} - Weight vs Activation Distributions', 
                     fontsize=14, fontweight='bold')
        
        # 1. Histogram comparison
        ax1 = axes[0, 0]
        ax1.hist(all_weights, bins=100, alpha=0.6, label='Weights', 
                 density=True, color='blue', edgecolor='black')
        ax1.hist(all_activations, bins=100, alpha=0.6, label='Activations', 
                 density=True, color='red', edgecolor='black')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution Comparison (Full Range)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Log-scale histogram for better visibility
        ax2 = axes[0, 1]
        ax2.hist(all_weights, bins=100, alpha=0.6, label='Weights', 
                 density=True, color='blue', edgecolor='black')
        ax2.hist(all_activations, bins=100, alpha=0.6, label='Activations', 
                 density=True, color='red', edgecolor='black')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Density (log scale)')
        ax2.set_yscale('log')
        ax2.set_title('Distribution Comparison (Log Scale)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Box plot comparison
        ax3 = axes[1, 0]
        bp = ax3.boxplot([all_weights, all_activations], 
                         labels=['Weights', 'Activations'],
                         patch_artist=True,
                         showfliers=False)  # Hide outliers for clarity
        bp['boxes'][0].set_facecolor('blue')
        bp['boxes'][1].set_facecolor('red')
        ax3.set_ylabel('Value')
        ax3.set_title('Box Plot Comparison (outliers hidden)')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Q-Q plot style comparison of percentiles
        ax4 = axes[1, 1]
        percentiles = np.linspace(0, 100, 101)
        weight_percentiles = np.percentile(all_weights, percentiles)
        activation_percentiles = np.percentile(all_activations, percentiles)
        ax4.scatter(weight_percentiles, activation_percentiles, 
                   alpha=0.5, s=20, color='purple')
        
        # Add diagonal line for reference
        min_val = min(weight_percentiles.min(), activation_percentiles.min())
        max_val = max(weight_percentiles.max(), activation_percentiles.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 
                'k--', alpha=0.5, label='y=x (same distribution)')
        
        ax4.set_xlabel('Weight Percentiles')
        ax4.set_ylabel('Activation Percentiles')
        ax4.set_title('Percentile-Percentile Plot')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            plot_path = os.path.join(save_dir, f'{layer_name}_distribution.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Plot saved: {plot_path}")
        
        plt.show()
        plt.close()
    
    # ====================================================================
    # CREATE SUMMARY TABLE
    # ====================================================================
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    df = pd.DataFrame(summary_data)
    
    # Format for display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.6f}'.format)
    
    print(df.to_string(index=False))
    
    if save_dir:
        csv_path = os.path.join(save_dir, 'weight_activation_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Summary saved: {csv_path}")
    
    return df

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

def main_v2():
    # Load data
    activation_file = os.path.join(
        RESULTS_DIR, 
        "activations", 
        "meta-llama_Llama-3.2-3B", 
        "custom_imc_key", 
        "activations.pkl"
    )
    
    if not os.path.exists(activation_file):
        print(f"❌ File not found: {activation_file}")
        return
    
    print(f"Loading data from: {activation_file}")
    data = data_loader(activation_file)
    print("✓ Data loaded successfully\n")
    
    # Create output directory
    output_dir = os.path.join(RESULTS_DIR, "weight_activation_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run analysis
    summary_df = analyze_weight_activation_distributions(data, save_dir=output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")

def main_domain_consistency():
    # Load paired datasets for domain consistency analysis
    model = "meta-llama_Llama-3.2-3B-Instruct"
    topic = "EV_ITALY"
    prompt_token = 92

    unpruned_file = os.path.join(
        RESULTS_DIR, 
        "knowledge_drift", 
        model,
        "EV_ITALY_0.0_prune_rel_gen700", 
        "activations.pkl"
    )
    
    pruned_file = os.path.join(
        RESULTS_DIR, 
        "knowledge_drift", 
        model, 
        "EV_ITALY_50.0_prune100_rel_gen700", 
        "activations.pkl"
    )

    pruned_file_0 = os.path.join(
        RESULTS_DIR, 
        "knowledge_drift", 
        model, 
        "EV_ITALY_50.0_prune50_rel_gen700", 
        "activations.pkl"
    )

    pruned_file_1 = os.path.join(
        RESULTS_DIR, 
        "knowledge_drift", 
        model, 
        "EV_ITALY_50.0_prune100_rel200_gen700", 
        "activations.pkl"
    )

    pruned_file_2 = os.path.join(
        RESULTS_DIR, 
        "knowledge_drift", 
        model, 
        "EV_ITALY_50.0_prune100_rel300_gen700", 
        "activations.pkl"
    )

    pruned_file_3 = os.path.join(
        RESULTS_DIR, 
        "knowledge_drift", 
        model, 
        "EV_ITALY_50.0_prune100_rel400_gen700", 
        "activations.pkl"
    )

    pruned_file_4 = os.path.join(
        RESULTS_DIR, 
        "knowledge_drift", 
        model, 
        "EV_ITALY_50.0_prune100_rel500_gen700", 
        "activations.pkl"
    )

    pruned_file_5 = os.path.join(
        RESULTS_DIR, 
        "knowledge_drift", 
        model, 
        "EV_ITALY_50.0_prune100_rel600_gen700", 
        "activations.pkl"
    )

    pruned_file_6 = os.path.join(
        RESULTS_DIR, 
        "knowledge_drift", 
        model, 
        "EV_ITALY_50.0_prune200_rel_gen700", 
        "activations.pkl"
    )

    topic_dir = os.path.join(DOMAIN_CONSISTENCY_DIR, model, topic)
    os.makedirs(topic_dir, exist_ok=True)
    
    paths_file = os.path.join(topic_dir, "source_paths.txt")
    
    with open(paths_file, 'w') as f:
        f.write("Source Data Paths\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Unpruned: {unpruned_file}\n")
        f.write(f"Pruned:   {pruned_file}\n")
    
    print(f"Source paths saved to: {paths_file}")

    # pruned_file_2 = os.path.join(
    #     RESULTS_DIR, 
    #     "knowledge_drift", 
    #     model, 
    #     "custom_50.0_prunes100_rel200_EV_ITALY", 
    #     "activations.pkl"
    # )

    # pruned_file_3 = os.path.join(
    #     RESULTS_DIR, 
    #     "knowledge_drift", 
    #     model, 
    #     "custom_50.0_prunes100_rel300_EV_ITALY", 
    #     "activations.pkl"
    # )
    
    if not os.path.exists(unpruned_file) or not os.path.exists(pruned_file):
        print(f"❌ One or both dataset files not found.")
        return
    
    print(f"Loading Dataset A from: {unpruned_file}")
    data_unpruned = data_loader(unpruned_file)
    print("✓ Dataset A loaded successfully\n")
    
    print(f"Loading Dataset B from: {pruned_file_0}")
    data_pruned_50 = data_loader(pruned_file_0)
    print("✓ Dataset B loaded successfully\n")

    print(f"Loading Dataset B from: {pruned_file}")
    data_pruned_100 = data_loader(pruned_file)
    print("✓ Dataset B loaded successfully\n")

    print(f"Loading Dataset C from: {pruned_file_1}")
    data_pruned_100_rel200 = data_loader(pruned_file_1)
    print("✓ Dataset C loaded successfully\n")

    print(f"Loading Dataset D from: {pruned_file_2}")
    data_pruned_100_rel300 = data_loader(pruned_file_2)
    print("✓ Dataset D loaded successfully\n")

    print(f"Loading Dataset B from: {pruned_file_3}")
    data_pruned_100_rel400 = data_loader(pruned_file_3)
    print("✓ Dataset B loaded successfully\n")

    print(f"Loading Dataset C from: {pruned_file_4}")
    data_pruned_100_rel500 = data_loader(pruned_file_4)
    print("✓ Dataset C loaded successfully\n")

    print(f"Loading Dataset D from: {pruned_file_5}")
    data_pruned_100_rel600 = data_loader(pruned_file_5)
    print("✓ Dataset D loaded successfully\n")

    print(f"Loading Dataset B from: {pruned_file_6}")
    data_pruned_200 = data_loader(pruned_file_6)
    print("✓ Dataset B loaded successfully\n")
    
    print("\n" + "="*80)
    print("BEFORE FLATTENING")
    print("="*80)
    print(f"Type: {type(data_unpruned['pre_ln2_activations'])}")
    print(f"Layers: {list(data_unpruned['pre_ln2_activations'].keys())[:5]}...")
    print(f"Layer 0 list length: {len(data_unpruned['pre_ln2_activations']['layer_0'])}")
    print(f"Layer 0, element 0 shape: {data_unpruned['pre_ln2_activations']['layer_0'][0].shape}")
    print(f"Layer 0, element 1 shape: {data_unpruned['pre_ln2_activations']['layer_0'][1].shape}")
    print(f"Layer 0, element -1 shape: {data_unpruned['pre_ln2_activations']['layer_0'][-1].shape}")
    
    # Flatten activation structures
    print("\n" + "="*80)
    print("FLATTENING ACTIVATION STRUCTURES")
    print("="*80)
    data_unpruned_flattened = {
        key: flatten_activation_structure(activations) 
        for key, activations in data_unpruned.items() 
        if isinstance(activations, dict) and any('layer' in k for k in activations.keys())
    }
    data_pruned_50_flattened = {
        key: flatten_activation_structure(activations) 
        for key, activations in data_pruned_50.items() 
        if isinstance(activations, dict) and any('layer' in k for k in activations.keys())
    }
    data_pruned_100_flattened = {
        key: flatten_activation_structure(activations) 
        for key, activations in data_pruned_100.items() 
        if isinstance(activations, dict) and any('layer' in k for k in activations.keys())
    }
    data_pruned_100_rel200_flattened = {
        key: flatten_activation_structure(activations) 
        for key, activations in data_pruned_100_rel200.items() 
        if isinstance(activations, dict) and any('layer' in k for k in activations.keys())
    }
    data_pruned_100_rel300_flattened = {
        key: flatten_activation_structure(activations) 
        for key, activations in data_pruned_100_rel300.items() 
        if isinstance(activations, dict) and any('layer' in k for k in activations.keys())
    }
    data_pruned_100_rel400_flattened = {
        key: flatten_activation_structure(activations) 
        for key, activations in data_pruned_100_rel400.items() 
        if isinstance(activations, dict) and any('layer' in k for k in activations.keys())
    }
    data_pruned_100_rel500_flattened = {
        key: flatten_activation_structure(activations) 
        for key, activations in data_pruned_100_rel500.items() 
        if isinstance(activations, dict) and any('layer' in k for k in activations.keys())
    }
    data_pruned_100_rel600_flattened = {
        key: flatten_activation_structure(activations) 
        for key, activations in data_pruned_100_rel600.items() 
        if isinstance(activations, dict) and any('layer' in k for k in activations.keys())
    }
    data_pruned_200_flattened = {
        key: flatten_activation_structure(activations) 
        for key, activations in data_pruned_200.items() 
        if isinstance(activations, dict) and any('layer' in k for k in activations.keys())
    }
    
    print("\n" + "="*80)
    print("AFTER FLATTENING")
    print("="*80)
    print(f"Layer 0 list length: {len(data_unpruned_flattened['pre_ln2_activations']['layer_0'])}")
    print(f"Layer 0, element 0 shape: {data_unpruned_flattened['pre_ln2_activations']['layer_0'][0].shape}")
    print(f"Layer 0, element 10 shape: {data_unpruned_flattened['pre_ln2_activations']['layer_0'][10].shape}")
    print(f"Layer 0, element -1 shape: {data_unpruned_flattened['pre_ln2_activations']['layer_0'][-1].shape}")
    print(f"✓ All elements now have shape (1, embed_dim)")
    
    # Update data dictionaries with flattened versions
    data_unpruned.update(data_unpruned_flattened)
    data_pruned_50.update(data_pruned_50_flattened)
    data_pruned_100.update(data_pruned_100_flattened)
    data_pruned_100_rel200.update(data_pruned_100_rel200_flattened)
    data_pruned_100_rel300.update(data_pruned_100_rel300_flattened)
    data_pruned_100_rel400.update(data_pruned_100_rel400_flattened)
    data_pruned_100_rel500.update(data_pruned_100_rel500_flattened)
    data_pruned_100_rel600.update(data_pruned_100_rel600_flattened)
    data_pruned_200.update(data_pruned_200_flattened)
    
    # data_c.update(data_c_flattened)
    # data_d.update(data_d_flattened)
# ------------------------------------------------------------------------
    # CENTER-TO-TOKEN ANALYSIS
# ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("ANALYZING DATA_UNPRUNED (Center-to-Token)")
    print("="*80)
    compare_c_t_Consistency(model, data_unpruned, embed="pre_ln2_activations", data_name="data_unpruned", topic=topic, masking_point=100+prompt_token, layer_num=0)
    compare_c_t_Consistency(model, data_unpruned, embed="pre_ln2_activations", data_name="data_unpruned", topic=topic, masking_point=100+prompt_token, layer_num=17)
    compare_c_t_Consistency(model, data_unpruned, embed="pre_ln2_activations", data_name="data_unpruned", topic=topic, masking_point=100+prompt_token, layer_num=-1)
    compare_c_t_Consistency(model, data_unpruned, embed="post_attn_oproj_activations", data_name="data_unpruned", topic=topic, masking_point=100+prompt_token, layer_num=0)
    compare_c_t_Consistency(model, data_unpruned, embed="post_attn_oproj_activations", data_name="data_unpruned", topic=topic, masking_point=100+prompt_token, layer_num=17)
    compare_c_t_Consistency(model, data_unpruned, embed="post_attn_oproj_activations", data_name="data_unpruned", topic=topic, masking_point=100+prompt_token, layer_num=-1)

    print("\n" + "="*80)
    print("ANALYZING DATA_PRUNED_50 (Center-to-Token)")
    print("="*80)
    compare_c_t_Consistency(model, data_pruned_50, embed="pre_ln2_activations", data_name="data_pruned_50", topic=topic, masking_point=50+prompt_token, layer_num=0)
    compare_c_t_Consistency(model, data_pruned_50, embed="pre_ln2_activations", data_name="data_pruned_50", topic=topic, masking_point=50+prompt_token, layer_num=17)
    compare_c_t_Consistency(model, data_pruned_50, embed="pre_ln2_activations", data_name="data_pruned_50", topic=topic, masking_point=50+prompt_token, layer_num=-1)
    compare_c_t_Consistency(model, data_pruned_50, embed="post_attn_oproj_activations", data_name="data_pruned_50", topic=topic, masking_point=50+prompt_token, layer_num=0)
    compare_c_t_Consistency(model, data_pruned_50, embed="post_attn_oproj_activations", data_name="data_pruned_50", topic=topic, masking_point=50+prompt_token, layer_num=17)
    compare_c_t_Consistency(model, data_pruned_50, embed="post_attn_oproj_activations", data_name="data_pruned_50", topic=topic, masking_point=50+prompt_token, layer_num=-1)

    print("\n" + "="*80)
    print("ANALYZING DATA_PRUNED_100 (Center-to-Token)")
    print("="*80)
    compare_c_t_Consistency(model, data_pruned_100, embed="pre_ln2_activations", data_name="data_pruned_100", topic=topic, masking_point=100+prompt_token, layer_num=0)
    compare_c_t_Consistency(model, data_pruned_100, embed="pre_ln2_activations", data_name="data_pruned_100", topic=topic, masking_point=100+prompt_token, layer_num=17)
    compare_c_t_Consistency(model, data_pruned_100, embed="pre_ln2_activations", data_name="data_pruned_100", topic=topic, masking_point=100+prompt_token, layer_num=-1)
    compare_c_t_Consistency(model, data_pruned_100, embed="post_attn_oproj_activations", data_name="data_pruned_100", topic=topic, masking_point=100+prompt_token, layer_num=0)
    compare_c_t_Consistency(model, data_pruned_100, embed="post_attn_oproj_activations", data_name="data_pruned_100", topic=topic, masking_point=100+prompt_token, layer_num=17)
    compare_c_t_Consistency(model, data_pruned_100, embed="post_attn_oproj_activations", data_name="data_pruned_100", topic=topic, masking_point=100+prompt_token, layer_num=-1)

    print("\n" + "="*80)
    print("ANALYZING DATA_PRUNED_100_REL200 (Center-to-Token)")
    print("="*80)
    compare_c_t_Consistency(model, data_pruned_100_rel200, embed="pre_ln2_activations", data_name="data_pruned_100_rel200", topic=topic, masking_point=100+prompt_token, layer_num=0)
    compare_c_t_Consistency(model, data_pruned_100_rel200, embed="pre_ln2_activations", data_name="data_pruned_100_rel200", topic=topic, masking_point=100+prompt_token, layer_num=17)
    compare_c_t_Consistency(model, data_pruned_100_rel200, embed="pre_ln2_activations", data_name="data_pruned_100_rel200", topic=topic, masking_point=100+prompt_token, layer_num=-1)
    compare_c_t_Consistency(model, data_pruned_100_rel200, embed="post_attn_oproj_activations", data_name="data_pruned_100_rel200", topic=topic, masking_point=100+prompt_token, layer_num=0)
    compare_c_t_Consistency(model, data_pruned_100_rel200, embed="post_attn_oproj_activations", data_name="data_pruned_100_rel200", topic=topic, masking_point=100+prompt_token, layer_num=17)
    compare_c_t_Consistency(model, data_pruned_100_rel200, embed="post_attn_oproj_activations", data_name="data_pruned_100_rel200", topic=topic, masking_point=100+prompt_token, layer_num=-1)

# ------------------------------------------------------------------------
    # CENTER-TO-WINDOW ANALYSIS
# ------------------------------------------------------------------------
    # print("\n" + "="*80)
    # print("ANALYZING DATA_A (Center-to-Window)")
    # print("="*80)
    # compare_c_w_Consistency(model, data_unpruned, embed="pre_ln2_activations", data_name="data_unpruned", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=17)
    # compare_c_w_Consistency(model, data_unpruned, embed="pre_ln2_activations", data_name="data_unpruned", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=0)
    # compare_c_w_Consistency(model, data_unpruned, embed="pre_ln2_activations", data_name="data_unpruned", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=-1)
    
    # # compare_c_w_Consistency(model, data_unpruned, embed="post_layer_activations", data_name="data_unpruned", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=-1)

    # compare_c_w_Consistency(model, data_unpruned, embed="post_attn_oproj_activations", data_name="data_unpruned", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=0)
    # compare_c_w_Consistency(model, data_unpruned, embed="post_attn_oproj_activations", data_name="data_unpruned", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=17)
    # compare_c_w_Consistency(model, data_unpruned, embed="post_attn_oproj_activations", data_name="data_unpruned", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=-1)


    # print("\n" + "="*80)
    # print("ANALYZING DATA_B (Center-to-Window)")
    # print("="*80)
    # compare_c_w_Consistency(model, data_pruned_50, embed="pre_ln2_activations", data_name="data_pruned_50", topic=topic, masking_point=50+prompt_token, window_size=10, layer_num=0)
    # compare_c_w_Consistency(model, data_pruned_50, embed="pre_ln2_activations", data_name="data_pruned_50", topic=topic, masking_point=50+prompt_token, window_size=10, layer_num=17)
    # compare_c_w_Consistency(model, data_pruned_50, embed="pre_ln2_activations", data_name="data_pruned_50", topic=topic, masking_point=50+prompt_token, window_size=10, layer_num=-1)

    # # compare_c_w_Consistency(model, data_pruned_50, embed="post_layer_activations", data_name="data_pruned_50", topic=topic, masking_point=50+prompt_token, window_size=10, layer_num=-1)
    # compare_c_w_Consistency(model, data_pruned_50, embed="post_attn_oproj_activations", data_name="data_pruned_50", topic=topic, masking_point=50+prompt_token, window_size=10, layer_num=-1)
    # compare_c_w_Consistency(model, data_pruned_50, embed="post_attn_oproj_activations", data_name="data_pruned_50", topic=topic, masking_point=50+prompt_token, window_size=10, layer_num=0)
    # compare_c_w_Consistency(model, data_pruned_50, embed="post_attn_oproj_activations", data_name="data_pruned_50", topic=topic, masking_point=50+prompt_token, window_size=10, layer_num=17)

    # # Run center-to-window analysis for data_b
    # print("\n" + "="*80)
    # print("ANALYZING DATA_B (Center-to-Window)")
    # print("="*80)
    # compare_c_w_Consistency(model, data_pruned_100, embed="pre_ln2_activations", data_name="data_pruned_100", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=0)
    # compare_c_w_Consistency(model, data_pruned_100, embed="pre_ln2_activations", data_name="data_pruned_100", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=17)
    # compare_c_w_Consistency(model, data_pruned_100, embed="pre_ln2_activations", data_name="data_pruned_100", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=-1)

    # # compare_c_w_Consistency(model, data_pruned_100, embed="post_layer_activations", data_name="data_pruned_100", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=-1)

    # compare_c_w_Consistency(model, data_pruned_100, embed="post_attn_oproj_activations", data_name="data_pruned_100", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=0)
    # compare_c_w_Consistency(model, data_pruned_100, embed="post_attn_oproj_activations", data_name="data_pruned_100", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=17)
    # compare_c_w_Consistency(model, data_pruned_100, embed="post_attn_oproj_activations", data_name="data_pruned_100", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=-1)

    # print("ANALYZING DATA_C (Center-to-Window)")
    # print("="*80)
    # compare_c_w_Consistency(model, data_pruned_100_rel200, embed="pre_ln2_activations", data_name="data_pruned_100_rel200", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=0)
    # compare_c_w_Consistency(model, data_pruned_100_rel200, embed="pre_ln2_activations", data_name="data_pruned_100_rel200", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=17)
    # compare_c_w_Consistency(model, data_pruned_100_rel200, embed="pre_ln2_activations", data_name="data_pruned_100_rel200", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=-1)
    # # compare_c_w_Consistency(model, data_pruned_100_rel200, embed="post_layer_activations", data_name="data_pruned_100_rel200", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=-1)

    # compare_c_w_Consistency(model, data_pruned_100_rel200, embed="post_attn_oproj_activations", data_name="data_pruned_100_rel200", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=0)
    # compare_c_w_Consistency(model, data_pruned_100_rel200, embed="post_attn_oproj_activations", data_name="data_pruned_100_rel200", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=17)
    # compare_c_w_Consistency(model, data_pruned_100_rel200, embed="post_attn_oproj_activations", data_name="data_pruned_100_rel200", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=-1)
    
    
    # print("\n" + "="*80)
    # print("ANALYZING DATA_D (Center-to-Window)")
    # print("="*80)
    # compare_c_w_Consistency(model, data_pruned_100_rel300, embed="pre_ln2_activations", data_name="data_pruned_100_rel300", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=0)
    # compare_c_w_Consistency(model, data_pruned_100_rel300, embed="pre_ln2_activations", data_name="data_pruned_100_rel300", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=17)

    # # compare_c_w_Consistency(model, data_pruned_100_rel300, embed="post_layer_activations", data_name="data_pruned_100_rel300", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=-1)
    
    # compare_c_w_Consistency(model, data_pruned_100_rel300, embed="post_attn_oproj_activations", data_name="data_pruned_100_rel300", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=0)
    # compare_c_w_Consistency(model, data_pruned_100_rel300, embed="post_attn_oproj_activations", data_name="data_pruned_100_rel300", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=17)

    # print("\n" + "="*80)
    # print("ANALYZING DATA_E (Center-to-Window)")
    # print("="*80)
    # compare_c_w_Consistency(model, data_pruned_100_rel400, embed="pre_ln2_activations", data_name="data_pruned_100_rel400", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=0)
    # compare_c_w_Consistency(model, data_pruned_100_rel400, embed="pre_ln2_activations", data_name="data_pruned_100_rel400", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=17)

    # # compare_c_w_Consistency(model, data_pruned_100_rel400, embed="post_layer_activations", data_name="data_pruned_100_rel400", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=-1)
    
    # compare_c_w_Consistency(model, data_pruned_100_rel400, embed="post_attn_oproj_activations", data_name="data_pruned_100_rel400", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=0)
    # compare_c_w_Consistency(model, data_pruned_100_rel400, embed="post_attn_oproj_activations", data_name="data_pruned_100_rel400", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=17)
    
    # print("\n" + "="*80)
    # print("ANALYZING DATA_F (Center-to-Window)")
    # print("="*80)
    # compare_c_w_Consistency(model, data_pruned_100_rel500, embed="pre_ln2_activations", data_name="data_pruned_100_rel500", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=0)
    # compare_c_w_Consistency(model, data_pruned_100_rel500, embed="pre_ln2_activations", data_name="data_pruned_100_rel500", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=17)

    # # compare_c_w_Consistency(model, data_pruned_100_rel500, embed="post_layer_activations", data_name="data_pruned_100_rel500", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=-1)
    
    # compare_c_w_Consistency(model, data_pruned_100_rel500, embed="post_attn_oproj_activations", data_name="data_pruned_100_rel500", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=0)
    # compare_c_w_Consistency(model, data_pruned_100_rel500, embed="post_attn_oproj_activations", data_name="data_pruned_100_rel500", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=17)
    
    # print("\n" + "="*80)
    # print("ANALYZING DATA_G (Center-to-Window)")
    # print("="*80)
    # compare_c_w_Consistency(model, data_pruned_100_rel600, embed="pre_ln2_activations", data_name="data_pruned_100_rel600", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=-1)

    # # compare_c_w_Consistency(model, data_pruned_100_rel600, embed="post_layer_activations", data_name="data_pruned_100_rel600", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=-1)
    
    # compare_c_w_Consistency(model, data_pruned_100_rel600, embed="post_attn_oproj_activations", data_name="data_pruned_100_rel600", topic=topic, masking_point=100+prompt_token, window_size=10, layer_num=-1)
    
    # print("\n" + "="*80)
    # print("ANALYZING DATA_G (Center-to-Window)")
    # print("="*80)
    # compare_c_w_Consistency(model, data_pruned_200, embed="pre_ln2_activations", data_name="data_pruned_200", topic=topic, masking_point=200+prompt_token, window_size=10, layer_num=0)
    # compare_c_w_Consistency(model, data_pruned_200, embed="pre_ln2_activations", data_name="data_pruned_200", topic=topic, masking_point=200+prompt_token, window_size=10, layer_num=17)

    # # compare_c_w_Consistency(model, data_pruned_200, embed="post_layer_activations", data_name="data_pruned_200", topic=topic, masking_point=200+prompt_token, window_size=10, layer_num=-1)

    # compare_c_w_Consistency(model, data_pruned_200, embed="post_attn_oproj_activations", data_name="data_pruned_200", topic=topic, masking_point=200+prompt_token, window_size=10, layer_num=0)
    # compare_c_w_Consistency(model, data_pruned_200, embed="post_attn_oproj_activations", data_name="data_pruned_200", topic=topic, masking_point=200+prompt_token, window_size=10, layer_num=17)

def main_attn_study():
    model = "meta-llama_Llama-3.2-3B-Instruct"
    topic = "EV_ITALY_short"
    prompt_token = 22

    unpruned_file = os.path.join(
        RESULTS_DIR, 
        "knowledge_drift", 
        model,
        "EV_ITALY_0.0_prune_rel_gen700", 
        "activations.pkl"
    )
    
    pruned_file = os.path.join(
        RESULTS_DIR, 
        "knowledge_drift", 
        model, 
        "EV_ITALY_50.0_prune100_rel_gen700", 
        "activations.pkl"
    )


    topic_dir = os.path.join(DOMAIN_CONSISTENCY_DIR, model, topic)
    os.makedirs(topic_dir, exist_ok=True)
    
    paths_file = os.path.join(topic_dir, "source_paths.txt")
    
    with open(paths_file, 'w') as f:
        f.write("Source Data Paths\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Unpruned: {unpruned_file}\n")
        f.write(f"Pruned:   {pruned_file}\n")
    
    print(f"Source paths saved to: {paths_file}")

    
    if not os.path.exists(unpruned_file) or not os.path.exists(pruned_file):
        print(f"❌ One or both dataset files not found.")
        return
    
    print(f"Loading Dataset A from: {unpruned_file}")
    data_unpruned = data_loader(unpruned_file)
    print("✓ Dataset A loaded successfully\n")
    
    print(f"Loading Dataset B from: {pruned_file}")
    data_pruned = data_loader(pruned_file)
    print("✓ Dataset B loaded successfully\n")

    data_unpruned_flattened = {
        key: flatten_activation_structure(activations) 
        for key, activations in data_unpruned.items() 
        if isinstance(activations, dict) and any('layer' in k for k in activations.keys())
    }
    data_pruned_flattened = {
        key: flatten_activation_structure(activations) 
        for key, activations in data_pruned.items() 
        if isinstance(activations, dict) and any('layer' in k for k in activations.keys())
    }
    
    # Update data dictionaries with flattened versions
    data_unpruned.update(data_unpruned_flattened)
    data_pruned.update(data_pruned_flattened)

    # Save the token wise attn_oproj embedding  cosine euclidean dist, etc etc with the average attn_oproj for the specified number of tokens. Look at C_W_Consistency function for reference
    print("\n" + "="*80)
    print("ANALYZING UNPRUNED DATA (Attention Output Projection Token-wise)")
    print("="*80)
    compare_attn_oproj_tokenwise(model, data_unpruned, embed="post_attn_oproj_activations", data_name="unpruned_1", topic=topic, num_reference_tokens=1, layer_num=-1)
    
    print("\n" + "="*80)
    print("ANALYZING PRUNED DATA (Attention Output Projection Token-wise)")
    print("="*80)
    compare_attn_oproj_tokenwise(model, data_pruned, embed="post_attn_oproj_activations", data_name="pruned_100_1", topic=topic, num_reference_tokens=1, layer_num=-1)
    
if __name__ == '__main__':
    #main_v2()

    main_domain_consistency()

    #main_attn_study()