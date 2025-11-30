import sys
import os
import json
from collections import defaultdict
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
from util import data_loader
from scipy.spatial.distance import cosine
import pandas as pd
import matplotlib.pyplot as plt

# Global path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
NEURON_DIR = os.path.join(RESULTS_DIR, "neurons")
MAX_FORWARD_PROXY = os.path.join(NEURON_DIR, "maxProxy")
MEAN_FORWARD_PROXY = os.path.join(NEURON_DIR, "meanProxy")
STRONG_WEAK_NEURONS = os.path.join(NEURON_DIR, "strong&weak")

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(NEURON_DIR, exist_ok=True)
os.makedirs(MAX_FORWARD_PROXY, exist_ok=True)
os.makedirs(MEAN_FORWARD_PROXY, exist_ok=True)
os.makedirs(STRONG_WEAK_NEURONS, exist_ok=True)

# PURPOSE OF THIS FILE:
# The purpose of this file is to find out the neuron activations that are the most relevant to certain concepts.
# That means we will remove those neurons that are not contributing anything strongly in the final output token prediction.
# This file does not aim to compare between prompts for now.
# Instead, it tries to find the important neurons for each prompt individually.
# Later on, we can compare which neurons are common across different prompts for the same concept.
# Right now, we limit our only to one prompt, look at it for a couple of iterations and then find the important neurons.

# STRATEGY:
# A normal approach which I have thought is to keep only the most contributing neurons(at the output tokens).
# For that, I have already calculated a forward proxy of the Value part of the MLP2 in the output token space.
# Now we can either just keep the max of this forward proxy as a sign that the neuron is important.
# Or we can also take into account the general impact of the neuron across all tokens and then decide its importance.
# But eventually, what we want to see is some neurons, which are repetitive, strongly influential. We will preserve.
# We will trim down some % of all the neurons to make sure we dont keep all of them and also don't let the LLM go crazy.(Perplexity maybe?)

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

def check_activation_difference(activations1: defaultdict(list), activations2: defaultdict(list)) -> float:
    # Flatten the activations to 1D
    for layer_name, activations_list in activations2.items():
        #last_gen_activations = activations_list[-1]
        print(f"Layer: {layer_name}, Activation1 Shape: {len(activations1[layer_name])}, Activation2 Shape: {len(activations2[layer_name])}")
        for i in range(len(activations1[layer_name])):
            embed1 = activations1[layer_name][i][-1]    
            embed2 = activations2[layer_name][i][-1]
            similarity = cosine_similarity(embed1, embed2)
            print(f"  Forward Pass {i}, Cosine Similarity: {similarity:.6f}")

    return similarity

def check_neuron_distribution(all_datasets: list[Tuple[str, dict[str, Any]]], k: int=1500):
    
    strong_neurons = {}
    weakly_strong_neurons = {}
    weak_neurons = {}
    
    #dataset_name = all_datasets[0][0]  # Get the dataset name
    dataset = all_datasets[0][1]
    
    for layer_name, neuron_proxy_list in dataset["mlp2_forward_proxy"].items():
        neuron_proxy_list_max = np.max(np.abs(neuron_proxy_list), axis=1)
        neuron_proxy_list_mean = np.mean(neuron_proxy_list, axis=1)
        
        print(f"Layer: {layer_name}, Neuron Proxy Max Shape: {neuron_proxy_list_max.shape}")
        print(f"Layer: {layer_name}, Neuron Proxy Mean Shape: {neuron_proxy_list_mean.shape}") 
        
        # Plot max values
        plt.figure(figsize=(20, 6))
        plt.plot(neuron_proxy_list_max, linewidth=0.5, alpha=0.8)
        plt.title(f"Neuron Proxy Max Values - {layer_name}")
        plt.xlabel("Neuron Index")
        plt.ylabel("Max Value")
        plt.grid(True, alpha=0.3)
        save_path = os.path.join(MAX_FORWARD_PROXY, f"{layer_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Get topk neurons based on max values
        topk_indices_max = set(np.argsort(neuron_proxy_list_max)[-k:].tolist())
        
        # Plot mean values
        plt.figure(figsize=(20, 6))
        plt.plot(neuron_proxy_list_mean, linewidth=0.5, alpha=0.8)
        plt.title(f"Neuron Proxy Mean Values - {layer_name}")
        plt.xlabel("Neuron Index")
        plt.ylabel("Mean Value")
        plt.grid(True, alpha=0.3)
        save_path = os.path.join(MEAN_FORWARD_PROXY, f"{layer_name}_mean.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Get topk neurons based on mean values
        topk_indices_mean = set(np.argsort(neuron_proxy_list_mean)[-k:].tolist())
        
        # Strong neurons: appear in both max and mean topk
        strong_neurons[layer_name] = topk_indices_max.intersection(topk_indices_mean)
        
        # Weakly strong neurons: appear in only one of the topk sets
        weakly_strong_neurons[layer_name] = topk_indices_max.symmetric_difference(topk_indices_mean)
        
        # Weak neurons: don't appear in either topk set
        total_neurons = neuron_proxy_list_max.shape[0]
        all_neurons = set(range(total_neurons))
        weak_neurons[layer_name] = all_neurons - topk_indices_max - topk_indices_mean
    
    return strong_neurons, weakly_strong_neurons, weak_neurons

def check_neuron_consistency(all_datasets: list[Tuple[str, dict[str, Any]]], k: int=1500):
    strong_neuron_counts = defaultdict(lambda: defaultdict(int))  # layer -> neuron_id -> count
    weakly_strong_neuron_counts = defaultdict(lambda: defaultdict(int))
    weak_neuron_counts = defaultdict(lambda: defaultdict(int))

    for dataset_name, dataset in all_datasets:    
        for layer_name, activations_list in dataset["pre_mlp2_activations"].items():
            forward_proxy = dataset["mlp2_forward_proxy"][layer_name]
            neuron_proxy_list_max = np.max(np.abs(forward_proxy), axis=1)
            neuron_proxy_list_mean = np.mean(np.abs(forward_proxy), axis=1)
            
            last_gen_activations = np.abs(activations_list[-1])  # Shape: (num_tokens, embed_dim)
            
            last_gen_forward_proxy_max = last_gen_activations * neuron_proxy_list_max  # Shape: (num_tokens, embed_dim)
            last_gen_forward_proxy_mean = last_gen_activations * neuron_proxy_list_mean  # Shape: (num_tokens, embed_dim)
            
            # Process each token separately
            num_tokens = last_gen_forward_proxy_max.shape[0]
            total_neurons = last_gen_forward_proxy_max.shape[1]
            
            for token_idx in range(num_tokens):
                # Get topk neurons for this token based on max values
                topk_indices_max = set(np.argsort(last_gen_forward_proxy_max[token_idx])[-k:].tolist())
                
                # Get topk neurons for this token based on mean values
                topk_indices_mean = set(np.argsort(last_gen_forward_proxy_mean[token_idx])[-k:].tolist())
                
                # Strong neurons: appear in both max and mean topk
                strong = topk_indices_max.intersection(topk_indices_mean)
                for neuron_id in strong:
                    strong_neuron_counts[layer_name][neuron_id] += 1
                
                # Weakly strong neurons: appear in only one of the topk sets
                weakly_strong = topk_indices_max.symmetric_difference(topk_indices_mean)
                for neuron_id in weakly_strong:
                    weakly_strong_neuron_counts[layer_name][neuron_id] += 1
                
                # Weak neurons: don't appear in either topk set
                all_neurons = set(range(total_neurons))
                weak = all_neurons - topk_indices_max - topk_indices_mean
                for neuron_id in weak:
                    weak_neuron_counts[layer_name][neuron_id] += 1
    
    # Convert to regular dict for easier analysis
    strong_neuron_counts = {layer: dict(counts) for layer, counts in strong_neuron_counts.items()}
    weakly_strong_neuron_counts = {layer: dict(counts) for layer, counts in weakly_strong_neuron_counts.items()}
    weak_neuron_counts = {layer: dict(counts) for layer, counts in weak_neuron_counts.items()}
    
    return strong_neuron_counts, weakly_strong_neuron_counts, weak_neuron_counts

def analyze_neuron_statistics(strong_neuron_counts, weakly_strong_neuron_counts, weak_neuron_counts, all_datasets: list[Tuple[str, dict[str, Any]]]):
    """
    Returns:
        layer_stats: Statistics per layer
        globally_strong_neurons: Neurons strong in ALL tokens per layer
        globally_weak_neurons: Neurons weak in ALL tokens per layer
    """
    save_dir = STRONG_WEAK_NEURONS

    layer_stats = {}
    globally_strong_neurons = {}
    globally_weak_neurons = {}
    
    # Assuming all datasets have the same number of tokens for simplicity
    num_tokens = None
    for dataset_name, dataset in all_datasets:
        sample_layer = next(iter(dataset["pre_mlp2_activations"].values()))
        num_tokens = sample_layer[-1].shape[0]
        break

    # Get all layers
    # Ideally just one of these datastructures.keys will be sufficient to get all the layers but it might happen
    # that some layers have no strong neurons so these layers will never be created since we are using defaultdict.
    all_layers = set(strong_neuron_counts.keys()) | set(weakly_strong_neuron_counts.keys()) | set(weak_neuron_counts.keys())
    
    for layer_name in all_layers:
        # Count how many neurons were strong, weakly strong, weak at least once
        num_strong = len(strong_neuron_counts.get(layer_name, {}))
        num_weakly_strong = len(weakly_strong_neuron_counts.get(layer_name, {}))
        num_weak = len(weak_neuron_counts.get(layer_name, {}))
        
        layer_stats[layer_name] = {
            'num_strong_neurons': num_strong,
            'num_weakly_strong_neurons': num_weakly_strong,
            'num_weak_neurons': num_weak
        }
        
        # Find globally strong neurons (strong in ALL tokens)
        globally_strong = set()
        if layer_name in strong_neuron_counts:
            for neuron_id, count in strong_neuron_counts[layer_name].items():
                if count == num_tokens:
                    globally_strong.add(neuron_id)
        globally_strong_neurons[layer_name] = globally_strong
        
        # Find globally weak neurons (weak in ALL tokens)
        globally_weak = set()
        if layer_name in weak_neuron_counts:
            for neuron_id, count in weak_neuron_counts[layer_name].items():
                if count == num_tokens:
                    globally_weak.add(neuron_id)
        globally_weak_neurons[layer_name] = globally_weak
        
        # Add global counts to stats
        layer_stats[layer_name]['num_globally_strong'] = len(globally_strong)
        layer_stats[layer_name]['num_globally_weak'] = len(globally_weak)
    
    # Save layer statistics
    with open(os.path.join(save_dir, 'layer_statistics.json'), 'w') as f:
        json.dump(layer_stats, f, indent=2)
    
    # Save globally strong neurons (convert sets to lists for JSON)
    globally_strong_serializable = {layer: sorted(list(neurons)) for layer, neurons in globally_strong_neurons.items()}
    with open(os.path.join(save_dir, 'globally_strong_neurons.json'), 'w') as f:
        json.dump(globally_strong_serializable, f, indent=2)
    
    # Save globally weak neurons (convert sets to lists for JSON)
    globally_weak_serializable = {layer: sorted(list(neurons)) for layer, neurons in globally_weak_neurons.items()}
    with open(os.path.join(save_dir, 'globally_weak_neurons.json'), 'w') as f:
        json.dump(globally_weak_serializable, f, indent=2)
    
    # Save full neuron counts for detailed analysis
    with open(os.path.join(save_dir, 'strong_neuron_counts.json'), 'w') as f:
        json.dump(strong_neuron_counts, f, indent=2)
    
    with open(os.path.join(save_dir, 'weakly_strong_neuron_counts.json'), 'w') as f:
        json.dump(weakly_strong_neuron_counts, f, indent=2)
    
    with open(os.path.join(save_dir, 'weak_neuron_counts.json'), 'w') as f:
        json.dump(weak_neuron_counts, f, indent=2)
    
    return layer_stats, globally_strong_neurons, globally_weak_neurons

def main():
    activation_imc = "results/pruneNeurons/activations.pkl"
    activation_imc_pruned = "results/pruneNeurons/activations_prune.pkl"
    
    data_imc = data_loader(activation_imc)
    data_imc_pruned = data_loader(activation_imc_pruned)

    all_datasets = [
        ("IMC", data_imc)
    ]

    check_activation_difference(
        data_imc["pre_mlp2_activations"],
        data_imc_pruned["pre_mlp2_activations"]
    )

    check_activation_difference(
        data_imc["post_mlp2_activations"],
        data_imc_pruned["post_mlp2_activations"]
    )

    # Here we are extracting the information about the forward proxy of the neurons.
    #strong_neurons, weakly_strong_neurons, weak_neurons = check_neuron_distribution(all_datasets)
    
    # Next we should get the neuron activations to observe which neurons have been consistently firing.
    # Among those firing, find out which ones do carry any impact. We can use the forward proxy for that.
    # Finally, we can trim down the neurons based on that information.
    
    #strong_counts, weakly_strong_counts, weak_counts = check_neuron_consistency(all_datasets, k=1000)
    #layer_stats, globally_strong, globally_weak = analyze_neuron_statistics(strong_counts, weakly_strong_counts, weak_counts, all_datasets)
    
    # Quick overview of the structure of the datasets
    #structure_explainer(data_imc)
    #neuron_(all_datasets)

if __name__ == '__main__':
    main()