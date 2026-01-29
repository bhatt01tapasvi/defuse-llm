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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

def run_experiment(model, layer_num, keep_rate, masking_step, ema_decay, ranking_method, prune_strategy, 
                   generation, cache_dir, parent_exp_dir, verbose,
                   prompt_type, prompt_subject, prompt_length, custom_prompt_text,
                   eval_perplexity, ppl_datasets, ppl_subjects, ppl_max_samples,
                   eval_mmlu, mmlu_datasets, mmlu_shots, mmlu_max_samples,
                   eval_general_nlp, general_nlp_datasets, general_nlp_max_samples,
                   base_keep_rate=None, save_activations=False):
    """
    Run a single pruning experiment for one layer at one keep rate.
    """
    # Create prompt-specific subdirectory
    prompt_dir = os.path.join(parent_exp_dir, f"{prompt_type}_{prompt_subject}")
    
    # Create experiment-specific directory within prompt subdirectory
    if base_keep_rate is not None:
        exp_dir = os.path.join(prompt_dir, 
                              f"layer_{layer_num}_base_keep_{base_keep_rate}_target_keep_{keep_rate}")
    else:
        exp_dir = os.path.join(prompt_dir, f"layer_{layer_num}_keep_{keep_rate}")
    
    os.makedirs(exp_dir, exist_ok=True)
    
    # Construct layer_topk argument
    if base_keep_rate is not None:
        layer_topk = f"all:{base_keep_rate},{layer_num}:{keep_rate}"
    else:
        layer_topk = f"{layer_num}:{keep_rate}"
    
    # Build command
    cmd = [
        "python", "dynamicPrune.py",
        "--model", model,
        "--cache_dir", cache_dir,
        "--layer_topk", layer_topk,
        "--maskingStep", str(masking_step),
        "--ranking_method", ranking_method,
        "--prune_strategy", prune_strategy,
        "--generation", str(generation),
        "--prompt_type", prompt_type,
        "--prompt_subject", prompt_subject,
        "--save_res_dir", exp_dir
    ]
    
    # Add ema_decay if specified
    if ema_decay is not None:
        cmd.extend(["--ema_decay", str(ema_decay)])
    
    # Add optional prompt arguments
    if prompt_length is not None:
        cmd.extend(["--prompt_length", str(prompt_length)])
    if custom_prompt_text is not None:
        cmd.extend(["--custom_prompt_text", custom_prompt_text])
    
    # Add evaluation flags
    if eval_perplexity:
        cmd.append("--eval_perplexity")
        if ppl_datasets:
            cmd.extend(["--ppl_datasets"] + ppl_datasets)
        if ppl_subjects:
            cmd.extend(["--ppl_subjects"] + ppl_subjects)
        if ppl_max_samples is not None:
            cmd.extend(["--ppl_max_samples", str(ppl_max_samples)])
    
    if eval_mmlu:
        cmd.append("--eval_mmlu")
        if mmlu_datasets:
            cmd.extend(["--mmlu_datasets"] + mmlu_datasets)
        if mmlu_shots is not None:
            cmd.extend(["--mmlu_shots", str(mmlu_shots)])
        if mmlu_max_samples is not None:
            cmd.extend(["--mmlu_max_samples", str(mmlu_max_samples)])
    
    if eval_general_nlp:
        cmd.append("--eval_general_nlp")
        if general_nlp_datasets:
            cmd.extend(["--general_nlp_datasets"] + general_nlp_datasets)
        if general_nlp_max_samples is not None:
            cmd.extend(["--general_nlp_max_samples", str(general_nlp_max_samples)])
    
    print(f"\n{'='*80}")
    if base_keep_rate is not None:
        print(f"Running MARGINAL ANALYSIS: Layer {layer_num}")
        print(f"  Base keep rate (all layers): {base_keep_rate}")
        print(f"  Target keep rate (layer {layer_num}): {keep_rate}")
    else:
        print(f"Running SINGLE LAYER: Layer {layer_num} | Keep Rate: {keep_rate}")
    print(f"Prompt: {prompt_type}_{prompt_subject}")
    print(f"Layer topk spec: {layer_topk}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Results will be saved to: {exp_dir}")
    print(f"{'='*80}\n")
    
    # Run the experiment
    try:
        subprocess.run(cmd, check=True)
        return True, exp_dir
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")
        print(f"Return code: {e.returncode}")
        return False, exp_dir

def run_group_experiment(model, layer_group, keep_rate, masking_step, ema_decay, ranking_method, prune_strategy,
                        generation, cache_dir, parent_exp_dir, verbose,
                        prompt_type, prompt_subject, prompt_length, custom_prompt_text,
                        eval_perplexity, ppl_datasets, ppl_subjects, ppl_max_samples,
                        eval_mmlu, mmlu_datasets, mmlu_shots, mmlu_max_samples,
                        eval_general_nlp, general_nlp_datasets, general_nlp_max_samples,
                        base_keep_rate=None, save_activations=False):
    """
    Run a pruning experiment for a group of layers at one keep rate.
    """
    # Create prompt-specific subdirectory
    prompt_dir = os.path.join(parent_exp_dir, f"{prompt_type}_{prompt_subject}")
    
    # Create experiment-specific directory within prompt subdirectory
    layer_str = "_".join(map(str, layer_group))
    if base_keep_rate is not None:
        exp_dir = os.path.join(prompt_dir, 
                              f"group_{layer_str}_base_{base_keep_rate}_target_{keep_rate}")
    else:
        exp_dir = os.path.join(prompt_dir, f"group_{layer_str}_keep_{keep_rate}")
    
    os.makedirs(exp_dir, exist_ok=True)
    
    # Construct layer_topk argument
    if base_keep_rate is not None:
        layer_specs = [f"all:{base_keep_rate}"]
        for layer_num in layer_group:
            layer_specs.append(f"{layer_num}:{keep_rate}")
        layer_topk = ",".join(layer_specs)
    else:
        layer_specs = [f"{layer_num}:{keep_rate}" for layer_num in layer_group]
        layer_topk = ",".join(layer_specs)
    
    # Build command
    cmd = [
        "python", "dynamicPrune.py",
        "--model", model,
        "--cache_dir", cache_dir,
        "--layer_topk", layer_topk,
        "--maskingStep", str(masking_step),
        "--ranking_method", ranking_method,
        "--prune_strategy", prune_strategy,
        "--generation", str(generation),
        "--prompt_type", prompt_type,
        "--prompt_subject", prompt_subject,
        "--save_res_dir", exp_dir
    ]
    
    # Add ema_decay if specified
    if ema_decay is not None:
        cmd.extend(["--ema_decay", str(ema_decay)])
    
    # Add optional prompt arguments
    if prompt_length is not None:
        cmd.extend(["--prompt_length", str(prompt_length)])
    if custom_prompt_text is not None:
        cmd.extend(["--custom_prompt_text", custom_prompt_text])
    
    # Add evaluation flags
    if eval_perplexity:
        cmd.append("--eval_perplexity")
        if ppl_datasets:
            cmd.extend(["--ppl_datasets"] + ppl_datasets)
        if ppl_subjects:
            cmd.extend(["--ppl_subjects"] + ppl_subjects)
        if ppl_max_samples is not None:
            cmd.extend(["--ppl_max_samples", str(ppl_max_samples)])
    
    if eval_mmlu:
        cmd.append("--eval_mmlu")
        if mmlu_datasets:
            cmd.extend(["--mmlu_datasets"] + mmlu_datasets)
        if mmlu_shots is not None:
            cmd.extend(["--mmlu_shots", str(mmlu_shots)])
        if mmlu_max_samples is not None:
            cmd.extend(["--mmlu_max_samples", str(mmlu_max_samples)])
    
    if eval_general_nlp:
        cmd.append("--eval_general_nlp")
        if general_nlp_datasets:
            cmd.extend(["--general_nlp_datasets"] + general_nlp_datasets)
        if general_nlp_max_samples is not None:
            cmd.extend(["--general_nlp_max_samples", str(general_nlp_max_samples)])
    
    print(f"\n{'='*80}")
    if base_keep_rate is not None:
        print(f"Running MARGINAL ANALYSIS: Layer Group {layer_group}")
        print(f"  Base keep rate (all layers): {base_keep_rate}")
        print(f"  Target keep rate (layers {layer_group}): {keep_rate}")
    else:
        print(f"Running LAYER GROUP: Layers {layer_group} | Keep Rate: {keep_rate}")
    print(f"Prompt: {prompt_type}_{prompt_subject}")
    print(f"Layer topk spec: {layer_topk}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Results will be saved to: {exp_dir}")
    print(f"{'='*80}\n")
    
    # Run the experiment
    try:
        subprocess.run(cmd, check=True)
        return True, exp_dir
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")
        print(f"Return code: {e.returncode}")
        return False, exp_dir

def run_baseline(model, generation, cache_dir, parent_exp_dir,
                prompt_type, prompt_length, custom_prompt_text,
                eval_perplexity, ppl_datasets, ppl_subjects, ppl_max_samples,
                eval_mmlu, mmlu_datasets, mmlu_shots, mmlu_max_samples,
                eval_general_nlp, general_nlp_datasets, general_nlp_max_samples,
                save_activations=False):
    """Run baseline experiment with no pruning."""
    # Create baseline directory (no prompt-specific subdirectory since no pruning)
    exp_dir = os.path.join(parent_exp_dir, "baseline")
    os.makedirs(exp_dir, exist_ok=True)
    
    cmd = [
        "python", "dynamicPrune.py",
        "--model", model,
        "--cache_dir", cache_dir,
        "--generation", str(generation),
        "--prompt_type", prompt_type,
        "--prompt_subject", "imc_key",
        "--save_res_dir", exp_dir
    ]
    
    # Add save_activations flag
    if save_activations:
        cmd.append("--save_activations")
    
    # Add optional prompt arguments
    if prompt_length is not None:
        cmd.extend(["--prompt_length", str(prompt_length)])
    if custom_prompt_text is not None:
        cmd.extend(["--custom_prompt_text", custom_prompt_text])
    
    # Add evaluation flags
    if eval_perplexity:
        cmd.append("--eval_perplexity")
        if ppl_datasets:
            cmd.extend(["--ppl_datasets"] + ppl_datasets)
        if ppl_subjects:
            cmd.extend(["--ppl_subjects"] + ppl_subjects)
        if ppl_max_samples is not None:
            cmd.extend(["--ppl_max_samples", str(ppl_max_samples)])
    
    if eval_mmlu:
        cmd.append("--eval_mmlu")
        if mmlu_datasets:
            cmd.extend(["--mmlu_datasets"] + mmlu_datasets)
        if mmlu_shots is not None:
            cmd.extend(["--mmlu_shots", str(mmlu_shots)])
        if mmlu_max_samples is not None:
            cmd.extend(["--mmlu_max_samples", str(mmlu_max_samples)])
    
    if eval_general_nlp:
        cmd.append("--eval_general_nlp")
        if general_nlp_datasets:
            cmd.extend(["--general_nlp_datasets"] + general_nlp_datasets)
        if general_nlp_max_samples is not None:
            cmd.extend(["--general_nlp_max_samples", str(general_nlp_max_samples)])
    
    print(f"\n{'='*80}")
    print(f"Running BASELINE (no pruning)")
    print(f"Results will be saved to: {exp_dir}")
    print(f"{'='*80}\n")
    
    try:
        subprocess.run(cmd, check=True)
        return True, exp_dir
    except subprocess.CalledProcessError as e:
        print(f"Error running baseline: {e}")
        print(f"Return code: {e.returncode}")
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
    
    # Model and basic config
    parser.add_argument('--model', type=str, default='gpt2', help='Model name/path')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--cache_dir', type=str, default='llm_weights', help='Cache directory for model weights')
    
    # Prompt configuration
    parser.add_argument('--prompt_type', type=str, default="custom", 
                       help='Type of prompt: mmlu, custom')
    parser.add_argument('--prompt_length', type=int, default=None,
                       help='Maximum prompt length in tokens. If None, uses full prompt.')
    parser.add_argument('--prompt_subject', nargs='+', type=str, default=["imc"],
                       help='Prompt subject names (can specify multiple). '
                            'Custom :: imc, imc2, imc3, imc_para, imc2_para, imc_word, '
                            'pizzas, pizzas_para, pizzas_word, actress, actress_para, actress_word, '
                            'astrophysics, astro_word, maths '
                            '| MMLU :: college_computer_science, machine_learning, electrical_engineering, '
                            'business_ethics, world_religions, prehistory, moral_disputes')
    parser.add_argument('--custom_prompt_text', type=str, default=None,
                       help='Custom prompt text (required if prompt_type=custom and prompt_subject is not specified)')
    
    # Generation and pruning
    parser.add_argument('--generation', type=int, default=150,
                       help='Number of tokens to generate for evaluation')
    parser.add_argument('--num_layers', type=int, default=None, 
                       help='Number of layers in model (auto-detected if not specified)')
    parser.add_argument('--keep_rates', nargs='+', type=float, 
                       default=[0.25, 0.5, 0.75],
                       help='Proportion of neurons to KEEP (e.g., 0.25 = keep 25%%, prune 75%%)')
    parser.add_argument('--base_keep_rate', type=float, default=None,
                       help='Base keep rate for all layers (enables marginal analysis mode). '
                            'If set, each experiment will keep this proportion in all layers, '
                            'then test varying keep rates on one layer at a time.')
    parser.add_argument('--masking_step', type=int, default=100,
                       help='Step at which to start masking')
    parser.add_argument('--ema_decay', type=float, default=None,
                       help='Decay factor for EMA (0.0 to 1.0). If None, uses L2 norm aggregation')
    parser.add_argument('--ranking_method', type=str, default='combined',
                       help='Method to rank neurons for pruning - max, mean, combined, magnitude')
    parser.add_argument('--prune_strategy', type=str, default='topk',
                       help='Pruning strategy - topk, auto')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output from NeuronDefuser')
    
    # Layer specification
    parser.add_argument('--layers', type=str, default=None,
                       help='Specific layers to test. Examples:\n'
                            '  "0,1,2,5-8" or "all" for individual layer mode\n'
                            '  "(1-5,7,10-13),(20-25,26,27)" for group mode (prune groups together)')
    
    # Experiment control
    parser.add_argument('--skip_baseline', action='store_true',
                       help='Skip baseline run')
    parser.add_argument('--parent_exp_dir', type=str, default=None,
                       help='Parent experiment directory (auto-generated if not specified)')
    
    # Evaluation: Perplexity
    parser.add_argument('--eval_perplexity', action='store_true', help='Evaluate perplexity on datasets')
    parser.add_argument('--ppl_datasets', nargs='+', default=["custom"], 
                       help='Datasets for perplexity eval. Options: custom, mmlu')
    parser.add_argument('--ppl_subjects', nargs='+', default=["imc"], 
                       help='Subjects for perplexity eval. Options: imc, anne_corpus, food_corpus, or MMLU subjects')
    parser.add_argument('--ppl_max_samples', type=int, default=None, 
                       help='Max samples for perplexity eval')
    
    # Evaluation: MMLU
    parser.add_argument('--eval_mmlu', action='store_true', help='Evaluate MMLU benchmark')
    parser.add_argument('--mmlu_datasets', nargs='+', 
                       default=["college_computer_science", "machine_learning", "electrical_engineering"], 
                       help='MMLU subjects to evaluate')
    parser.add_argument('--mmlu_shots', type=int, default=0, help='Number of shots for MMLU evaluation')
    parser.add_argument('--mmlu_max_samples', type=int, default=None, help='Max samples for MMLU eval')
    
    # Evaluation: General NLP
    parser.add_argument('--eval_general_nlp', action='store_true', help='Evaluate general NLP benchmark')
    parser.add_argument('--general_nlp_datasets', nargs='+', default=None, 
                       help='Datasets for general NLP eval. Options: boolq, rte, hellaswag, winogrande, arc_easy, arc_challenge, openbookqa')
    parser.add_argument('--general_nlp_max_samples', type=int, default=None, 
                       help='Max samples for general NLP eval')
    
    # Activation saving
    parser.add_argument('--save_activations', action='store_true', help='Save activations during model run')
    
    args = parser.parse_args()

    # Auto-detect number of layers if not specified
    if args.num_layers is None:
        print("Auto-detecting number of layers...")
        if 'gpt2' in args.model.lower():
            num_layers = 12
        elif 'llama' in args.model.lower() or 'Llama' in args.model:
            # Llama-3.1-8B has 32 layers
            num_layers = 32
        else:
            raise ValueError("Please specify --num_layers for this model")
    else:
        num_layers = args.num_layers

    is_group_mode, specified_layers = parse_layer_specification(args.layers, num_layers)
    
    # Generate experiment name
    if args.parent_exp_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if args.base_keep_rate:
            mode = f"marginal_{args.base_keep_rate}"
        else:
            mode = "single"
        if is_group_mode:
            mode += "_group"
        args.parent_exp_dir = f"{args.model.replace('/', '_')}_{mode}_{timestamp}"
    
    print(f"\n{'='*80}")
    print(f"LAYER-WISE PRUNING ANALYSIS")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Number of layers: {num_layers}")
    print(f"Mode: {'GROUP' if is_group_mode else 'INDIVIDUAL'}")
    print(f"Keep Rates: {args.keep_rates}")
    print(f"Masking Step: {args.masking_step}")
    print(f"EMA Decay: {args.ema_decay if args.ema_decay is not None else 'None (L2 norm)'}")
    print(f"Ranking Method: {args.ranking_method}")
    print(f"Prune Strategy: {args.prune_strategy}")
    print(f"Prompt Type: {args.prompt_type}")
    print(f"Prompt Subject: {args.prompt_subject}")
    print(f"Experiment Directory Name: {args.parent_exp_dir}")
    print(f"Evaluations: ", end="")
    evals = []
    if args.eval_perplexity:
        evals.append("Perplexity")
    if args.eval_mmlu:
        evals.append(f"MMLU ({args.mmlu_shots}-shot)")
    if args.eval_general_nlp:
        evals.append("General NLP")
    print(", ".join(evals) if evals else "None")
    print(f"{'='*80}\n")

    if is_group_mode:
        print(f"Testing layer groups:")
        for i, group in enumerate(specified_layers):
            print(f"  Group {i+1}: {group}")
        total_experiments = len(specified_layers) * len(args.keep_rates) * len(args.prompt_subject)
    else:
        print(f"Testing layers individually: {specified_layers}")
        total_experiments = len(specified_layers) * len(args.keep_rates) * len(args.prompt_subject)
    
    total_experiments += 0 if args.skip_baseline else 1  # Only 1 baseline run
    print(f"Total experiments: {total_experiments}\n")
    
    # Run baseline once (no pruning, independent of prompt subjects)
    if not args.skip_baseline:
        success, _ = run_baseline(
            model=args.model,
            generation=args.generation,
            cache_dir=args.cache_dir,
            parent_exp_dir=args.parent_exp_dir,
            prompt_type=args.prompt_type,
            prompt_length=args.prompt_length,
            custom_prompt_text=args.custom_prompt_text,
            eval_perplexity=args.eval_perplexity,
            ppl_datasets=args.ppl_datasets,
            ppl_subjects=args.ppl_subjects,
            ppl_max_samples=args.ppl_max_samples,
            eval_mmlu=args.eval_mmlu,
            mmlu_datasets=args.mmlu_datasets,
            mmlu_shots=args.mmlu_shots,
            mmlu_max_samples=args.mmlu_max_samples,
            eval_general_nlp=args.eval_general_nlp,
            general_nlp_datasets=args.general_nlp_datasets,
            general_nlp_max_samples=args.general_nlp_max_samples,
            save_activations=args.save_activations
        )
        if not success:
            print("WARNING: Baseline experiment failed!")
    
    # Run experiments for each layer and keep rate
    completed = 0

    if is_group_mode:
        # Group mode: test each group at each keep rate and each prompt subject
        for group_layers in specified_layers:
            for keep_rate in args.keep_rates:
                for prompt_subject in args.prompt_subject:
                    success, exp_dir = run_group_experiment(
                        model=args.model,
                        layer_group=group_layers,
                        keep_rate=keep_rate,
                        masking_step=args.masking_step,
                        ema_decay=args.ema_decay,
                        ranking_method=args.ranking_method,
                        prune_strategy=args.prune_strategy,
                        generation=args.generation,
                        cache_dir=args.cache_dir,
                        parent_exp_dir=args.parent_exp_dir,
                        verbose=args.verbose,
                        prompt_type=args.prompt_type,
                        prompt_subject=prompt_subject,
                        prompt_length=args.prompt_length,
                        custom_prompt_text=args.custom_prompt_text,
                        eval_perplexity=args.eval_perplexity,
                        ppl_datasets=args.ppl_datasets,
                        ppl_subjects=args.ppl_subjects,
                        ppl_max_samples=args.ppl_max_samples,
                        eval_mmlu=args.eval_mmlu,
                        mmlu_datasets=args.mmlu_datasets,
                        mmlu_shots=args.mmlu_shots,
                        mmlu_max_samples=args.mmlu_max_samples,
                        eval_general_nlp=args.eval_general_nlp,
                        general_nlp_datasets=args.general_nlp_datasets,
                        general_nlp_max_samples=args.general_nlp_max_samples,
                        base_keep_rate=args.base_keep_rate,
                        save_activations=args.save_activations
                    )
                    completed += 1
                    print(f"\nProgress: {completed}/{total_experiments} experiments completed\n")
    else:
        # Individual mode: test each layer separately at each keep rate and each prompt subject
        for layer_num in specified_layers:
            for keep_rate in args.keep_rates:
                for prompt_subject in args.prompt_subject:
                    success, exp_dir = run_experiment(
                        model=args.model,
                        layer_num=layer_num,
                        keep_rate=keep_rate,
                        masking_step=args.masking_step,
                        ema_decay=args.ema_decay,
                        ranking_method=args.ranking_method,
                        prune_strategy=args.prune_strategy,
                        generation=args.generation,
                        cache_dir=args.cache_dir,
                        parent_exp_dir=args.parent_exp_dir,
                        verbose=args.verbose,
                        prompt_type=args.prompt_type,
                        prompt_subject=prompt_subject,
                        prompt_length=args.prompt_length,
                        custom_prompt_text=args.custom_prompt_text,
                        eval_perplexity=args.eval_perplexity,
                        ppl_datasets=args.ppl_datasets,
                        ppl_subjects=args.ppl_subjects,
                        ppl_max_samples=args.ppl_max_samples,
                        eval_mmlu=args.eval_mmlu,
                        mmlu_datasets=args.mmlu_datasets,
                        mmlu_shots=args.mmlu_shots,
                        mmlu_max_samples=args.mmlu_max_samples,
                        eval_general_nlp=args.eval_general_nlp,
                        general_nlp_datasets=args.general_nlp_datasets,
                        general_nlp_max_samples=args.general_nlp_max_samples,
                        base_keep_rate=args.base_keep_rate,
                        save_activations=args.save_activations
                    )
                    completed += 1
                    print(f"\nProgress: {completed}/{total_experiments} experiments completed\n")

    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETED")
    print(f"{'='*80}")
    print(f"Results saved to: {args.parent_exp_dir}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()