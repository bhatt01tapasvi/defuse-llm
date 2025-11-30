#!/usr/bin/env python3
"""
Rigorous layer-wise pruning analysis script.
Tests the effect of pruning each layer individually at different pruning rates.
"""

import subprocess
import os
import json
import argparse
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "layer_analysis")

def run_experiment(model, layer_num, keep_rate, masking_step, generation, cache_dir, exp_name, base_keep_rate=None):
    """
    Run a single pruning experiment for one layer at one keep rate.
    
    Args:
        model: Model name/path
        layer_num: Layer number to keep
        keep_rate: Keep rate (proportion of neurons to keep, e.g., 0.25, 0.5, 0.75)
        masking_step: Step at which to start masking
        cache_dir: Cache directory for model weights
        exp_name: Experiment name for organizing results
    """
    # Create experiment-specific directory
    if base_keep_rate is not None:
        # Marginal analysis mode
        exp_dir = os.path.join(ANALYSIS_DIR, exp_name, 
                              f"layer_{layer_num}_base_keep_{base_keep_rate}_target_keep_{keep_rate}")
    else:
        # Single layer pruning mode
        exp_dir = os.path.join(ANALYSIS_DIR, exp_name, f"layer_{layer_num}_keep_{keep_rate}")
    
    os.makedirs(exp_dir, exist_ok=True)
    
    # Construct layer_topk argument
    if base_keep_rate is not None:
        # Apply base_keep_rate to all layers, then override the target layer
        layer_topk = f"all:{base_keep_rate},{layer_num}:{keep_rate}"
    else:
        # Only keep the target layer
        layer_topk = f"{layer_num}:{keep_rate}"
    
    # Build command
    cmd = [
        "python", "dynamicPrune.py",
        "--model", model,
        "--cache_dir", cache_dir,
        "--layer_topk", layer_topk,
        "--maskingStep", str(masking_step),
        "--generation", str(generation),  # Generate more tokens for better evaluation
        "--eval_perplexity",
        "--save_res_dir", exp_dir
    ]
    
    print(f"\n{'='*80}")
    if base_keep_rate is not None:
        print(f"Running MARGINAL ANALYSIS: Layer {layer_num}")
        print(f"  Base keep rate (all layers): {base_keep_rate}")
        print(f"  Target keep rate (layer {layer_num}): {keep_rate}")
    else:
        print(f"Running SINGLE LAYER: Layer {layer_num} | Keep Rate: {keep_rate}")
    print(f"Layer topk spec: {layer_topk}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Results will be saved to: {exp_dir}")
    print(f"{'='*80}\n")
    
    # Run the experiment
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True, exp_dir
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False, exp_dir

def run_group_experiment(model, layer_group, keep_rate, masking_step, generation, cache_dir, exp_name, base_keep_rate=None):
    """
    Run a pruning experiment for a group of layers at one keep rate.
    
    Args:
        model: Model name/path
        layer_group: List of layer numbers to keep together
        keep_rate: Keep rate (e.g., 0.25, 0.5, 0.75)
        masking_step: Step at which to start masking
        generation: Number of tokens to generate
        cache_dir: Cache directory for model weights
        exp_name: Experiment name for organizing results
        base_keep_rate: Optional base keep rate for all other layers
    """
    # Create experiment-specific directory
    layer_str = "_".join(map(str, layer_group))
    if base_keep_rate is not None:
        exp_dir = os.path.join(ANALYSIS_DIR, exp_name, 
                              f"group_{layer_str}_base_{base_keep_rate}_target_{keep_rate}")
    else:
        exp_dir = os.path.join(ANALYSIS_DIR, exp_name, f"group_{layer_str}_keep_{keep_rate}")
    
    os.makedirs(exp_dir, exist_ok=True)
    
    # Construct layer_topk argument
    if base_keep_rate is not None:
        # Apply base_keep_rate to all layers, then override the target group
        layer_specs = [f"all:{base_keep_rate}"]
        for layer_num in layer_group:
            layer_specs.append(f"{layer_num}:{keep_rate}")
        layer_topk = ",".join(layer_specs)
    else:
        # Only keep the target group of layers
        layer_specs = [f"{layer_num}:{keep_rate}" for layer_num in layer_group]
        layer_topk = ",".join(layer_specs)
    
    # Build command
    cmd = [
        "python", "dynamicPrune.py",
        "--model", model,
        "--cache_dir", cache_dir,
        "--layer_topk", layer_topk,
        "--maskingStep", str(masking_step),
        "--generation", str(generation),
        "--eval_perplexity",
        "--save_res_dir", exp_dir
    ]
    
    print(f"\n{'='*80}")
    if base_keep_rate is not None:
        print(f"Running MARGINAL ANALYSIS: Layer Group {layer_group}")
        print(f"  Base keep rate (all layers): {base_keep_rate}")
        print(f"  Target keep rate (layers {layer_group}): {keep_rate}")
    else:
        print(f"Running LAYER GROUP: Layers {layer_group} | Keep Rate: {keep_rate}")
    print(f"Layer topk spec: {layer_topk}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Results will be saved to: {exp_dir}")
    print(f"{'='*80}\n")
    
    # Run the experiment
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True, exp_dir
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False, exp_dir

def run_baseline(model, generation, cache_dir, exp_name):
    """Run baseline experiment with no pruning."""
    exp_dir = os.path.join(ANALYSIS_DIR, exp_name, "baseline")
    os.makedirs(exp_dir, exist_ok=True)
    
    cmd = [
        "python", "dynamicPrune.py",
        "--model", model,
        "--cache_dir", cache_dir,
        "--generation", str(generation),
        "--eval_perplexity",
        "--save_res_dir", exp_dir
    ]
    
    print(f"\n{'='*80}")
    print(f"Running BASELINE (no pruning)")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True, exp_dir
    except subprocess.CalledProcessError as e:
        print(f"Error running baseline: {e}")
        return False, exp_dir

def parse_layer_specification(layer_spec, num_layers):
    """
    Parse layer specification string into list of layers or groups.
    
    Args:
        layer_spec: String like "0,1,2,5-8" or "(1-5,7,10-13),(20-25,26,27)" or "all"
    
    Returns:
        tuple: (is_group_mode, result)
            - is_group_mode: True if group mode, False if individual mode
            - result: For individual mode: list of ints
                     For group mode: list of lists (each inner list is a group)
    """
    if layer_spec is None or layer_spec == "all":
        layers = list(range(num_layers))
        return False, layers
    
    # Check if it's group mode (contains parentheses)
    if '(' in layer_spec and ')' in layer_spec:
        is_group_mode = True
        groups = []
        
        # Extract content within parentheses
        import re
        group_matches = re.findall(r'\(([^)]+)\)', layer_spec)
        
        for group_str in group_matches:
            group_layers = []
            for part in group_str.split(','):
                part = part.strip()
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    group_layers.extend(range(start, end + 1))
                else:
                    group_layers.append(int(part))
            groups.append(group_layers)
        
        return is_group_mode, groups
    else:
        is_group_mode = False
        # Parse layer numbers for individual mode
        layers = []
        for part in layer_spec.split(','):
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                layers.extend(range(start, end + 1))
            else:
                layers.append(int(part))
        
        return is_group_mode, layers
    
def main():
    parser = argparse.ArgumentParser(description='Run layer-wise pruning analysis')
    parser.add_argument('--model', type=str, default='gpt2', help='Model name/path')
    parser.add_argument('--num_layers', type=int, default=None, 
                       help='Number of layers')
    parser.add_argument('--keep_rates', nargs='+', type=float, 
                       default=[0.25, 0.5, 0.75],
                       help='Proportion of neurons to KEEP (e.g., 0.25 = keep 25%%, prune 75%%)')
    parser.add_argument('--base_keep_rate', type=float, default=None,
                       help='Base keep rate for all layers (enables marginal analysis mode). '
                            'If set, each experiment will keep this proportion in all layers, '
                            'then test varying keep rates on one layer at a time.')
    parser.add_argument('--masking_step', type=int, default=100,
                       help='Step at which to start masking')
    parser.add_argument('--generation', type=int, default=150,
                       help='Number of tokens to generate for evaluation')
    parser.add_argument('--cache_dir', type=str, default='llm_weights',
                       help='Cache directory for model weights')
    parser.add_argument('--skip_baseline', action='store_true',
                       help='Skip baseline run')
    parser.add_argument('--layers', type=str, default=None,
                       help='Specific layers to test. Examples:\n'
                            '  "0,1,2,5-8" or "all" for individual layer mode\n'
                            '  "(1-5,7,10-13),(20-25,26,27)" for group mode (prune groups together)')
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name (auto-generated if not specified)')
    args = parser.parse_args()

    if args.num_layers is None:
        print("Auto-detecting number of layers...")
        # You might need to add a helper function to detect this
        # For now, use common defaults
        if 'gpt2' in args.model.lower():
            num_layers = 12
        elif 'llama' in args.model.lower():
            num_layers = 32
        else:
            raise ValueError("Please specify --num_layers for this model")
    else:
        num_layers = args.num_layers

    is_group_mode, specified_layers = parse_layer_specification(args.layers, num_layers)
    
    # Generate experiment name
    if args.exp_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if args.base_keep_rate:
            mode = f"marginal_{args.base_keep_rate}"
        else:
            mode = "single"
        if is_group_mode:
            mode += "_group"
        args.exp_name = f"{args.model.replace('/', '_')}_{mode}_{timestamp}"
    
    print(f"\n{'='*80}")
    print(f"LAYER-WISE PRUNING ANALYSIS")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Mode: {'GROUP' if is_group_mode else 'INDIVIDUAL'}")
    print(f"Keep Rates: {args.keep_rates}")
    print(f"Masking Step: {args.masking_step}")
    print(f"Experiment Name: {args.exp_name}")
    print(f"{'='*80}\n")

    if is_group_mode:
        print(f"Testing layer groups:")
        for i, group in enumerate(specified_layers):
            print(f"  Group {i+1}: {group}")
        total_experiments = len(specified_layers) * len(args.keep_rates)
    else:
        print(f"Testing layers individually: {specified_layers}")
        # In individual mode, we test each layer separately
        total_experiments = len(specified_layers) * len(args.keep_rates)
    
    total_experiments += 0 if args.skip_baseline else 1
    print(f"Total experiments: {total_experiments}\n")
    
    # Run baseline
    if not args.skip_baseline:
        success, _ = run_baseline(args.model, args.generation, args.cache_dir, args.exp_name)
        if not success:
            print("WARNING: Baseline experiment failed!")
    
    # Run experiments for each layer and keep rate
    completed = 0

    if is_group_mode:
        # Group mode: test each group at each keep rate
        for group_layers in specified_layers:
            for keep_rate in args.keep_rates:
                success, exp_dir = run_group_experiment(
                    model=args.model,
                    layer_group=group_layers,
                    keep_rate=keep_rate,  # Internal param name stays for now
                    masking_step=args.masking_step,
                    generation=args.generation,
                    cache_dir=args.cache_dir,
                    exp_name=args.exp_name,
                    base_keep_rate=args.base_keep_rate
                )
                completed += 1
                print(f"\nProgress: {completed}/{total_experiments} experiments completed\n")
    else:
        # Individual mode: test each layer separately
        for layer_num in specified_layers:
            for keep_rate in args.keep_rates:
                success, exp_dir = run_experiment(
                    model=args.model,
                    layer_num=layer_num,
                    keep_rate=keep_rate,
                    masking_step=args.masking_step,
                    generation=args.generation,
                    cache_dir=args.cache_dir,
                    exp_name=args.exp_name,
                    base_keep_rate=args.base_keep_rate
                )
                completed += 1
                print(f"\nProgress: {completed}/{total_experiments} experiments completed\n")

if __name__ == "__main__":
    main()