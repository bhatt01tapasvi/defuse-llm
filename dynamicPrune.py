import argparse
from collections import defaultdict
import os 
import numpy as np
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
from typing import Dict
import pickle
import matplotlib.pyplot as plt
from datasets import config
import time

from src.neuronDefuser import NeuronDefuser
from src.perplexity_utils import evaluate_on_datasets
from src.mmlu_utils import get_mmlu_prompt_concat
from src.general_nlp_utils import evaluate_general_nlp
from src.hook_setup import setup_hooks_gpt2, setup_hooks_llama

try:
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import handle_non_serializable
    LM_EVAL_AVAILABLE = True
except ImportError:
    LM_EVAL_AVAILABLE = False
    print("Warning: lm-evaluation-harness not installed. Install with: pip install lm-eval")

from huggingface_hub import login
login(token="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")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

######## Purpose of the file.
# The main aim of this file is to run the llm and capture all the data throughout it's lifetime.
# The modification can be executed here in pre_hook functions but that should be a single focused contribution in this file.
# Rest of the analysis will happen in separate files (existing in src).
########

# Print package versions and GPU info
print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )
    return model

def detect_model_type(model):
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return 'gpt2'
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return 'llama'
    else:
        raise ValueError("Unknown model architecture. Only GPT2 and LLaMA are supported.")

def parse_layer_topk(layer_spec: str, num_layers: int, intermediate_size: int) -> dict:
    """
    Parse layer topk specification string into a dictionary.
    
    Args:
        layer_spec: One of the following formats:
            - "all:0.6" - Apply 60% to all layers
            - "0,1,2:0.6" - Apply 60% to layers 0, 1, 2
            - "0:1000,1:1500,2:2000" - Specific neuron counts per layer
            - "0-5:0.6" - Apply 60% to layers 0 through 5
            - "0-5:0.6,6-11:0.8" - Different percentages for different ranges
        num_layers: Total number of layers in the model
        intermediate_size: Size of the intermediate layer (for percentage calculations)
    
    Returns:
        Dictionary mapping layer numbers to topk values
    """
    per_layer_topk = {}
    
    if layer_spec is None:
        return per_layer_topk
    
    # Split by comma to handle multiple specifications
    specs = layer_spec.split(',')
    
    for spec in specs:
        spec = spec.strip()
        if ':' not in spec:
            raise ValueError(f"Invalid layer spec format: {spec}. Expected format 'layer:value'")
        
        layer_part, value_part = spec.split(':', 1)
        layer_part = layer_part.strip()
        value_part = value_part.strip()
        
        # Parse the value (either percentage or absolute number)
        try:
            value = float(value_part)
            if 0 < value <= 1:
                # It's a percentage
                topk_value = int(value * intermediate_size)
            else:
                # It's an absolute number
                topk_value = int(value)
        except ValueError:
            raise ValueError(f"Invalid value: {value_part}. Must be a number.")
        
        # Parse the layer specification
        if layer_part == "all":
            # Apply to all layers
            for i in range(num_layers):
                per_layer_topk[i] = topk_value
        elif '-' in layer_part:
            # Range specification (e.g., "0-5")
            start, end = layer_part.split('-')
            start = int(start.strip())
            end = int(end.strip())
            for i in range(start, end + 1):
                if i < num_layers:
                    per_layer_topk[i] = topk_value
        else:
            # Single layer or will be processed as individual layers
            try:
                layer_num = int(layer_part)
                if layer_num < num_layers:
                    per_layer_topk[layer_num] = topk_value
            except ValueError:
                raise ValueError(f"Invalid layer specification: {layer_part}")
    
    return per_layer_topk

def initialize_prompt(tokenizer, device, max_seq_len: int, prompt_type="custom", prompt_length=None, prompt_subject="imc", custom_text=None):
    """
    Initialize a prompt based on type and length constraints.
    
    Args:
        tokenizer: Model tokenizer
        device: Device to place tensors on
        max_seq_len: Maximum sequence length supported by the model
        prompt_type: Type of prompt. Options:
            - "mmlu" (requires prompt_subject parameter)
            - "custom" (requires prompt subject or custom_text parameter)
        prompt_length: Maximum number of tokens. If None, uses full prompt (capped at max_seq_len).
                      If specified, truncates prompt to this length.
        prompt_subject: Prompt subject name (required if prompt_type="mmlu")
        custom_text: Custom text string (required if prompt_type="custom")
    
    Returns:
        tuple: (prompt_text, input_ids_tensor)
    """
    
    # Define all available custom_prompts
    custom_prompts = {
        # In-Memory Computing prompts
        "imc": "In-memory computing enables parallel processing by performing arithmetic operations directly within memory arrays, eliminating the need for data movement between processor and memory in the system.",
        
        "imc2": "Processing in memory is an emerging non-von Neumann computational paradigm whose key idea is to perform certain computational tasks in place in memory, thereby obviating the need to shuttle data back and forth between the processing and memory units.",
        
        "imc3": "In-memory computation works by eliminating all slow data accesses and relying exclusively on data stored in RAM. Overall computation performance is greatly improved by removing the latency commonly seen when accessing hard disk drives or SSDs.",
        
        "imc_para": "In-memory computing (IMC) is an emerging non-von Neumann computational paradigm that keeps alive the promise of achieving energy efficiencies on the order of one femtoJoule per operation in a computing system. The key idea is to perform certain computational tasks in place in memory, thereby obviating the need to shuttle data back and forth between the processing and memory units. The time and energy cost associated with this data movement is by far the most severe roadblock for modern computing systems. IMC is often achieved by exploiting the physical attributes of the memory devices, their array-level organization, etc. IMC has found application in a range of applications such as scientific computing, database query, machine learning etc. However, the most promising application for IMC is for efficient realization of deep neural networks (DNNs) that have revolutionized AI in recent years. A key challenge for DNNs is its computational inefficiency. In fact, the lack of sufficient compute power was one of the key factors that held back progress in the field for almost 30 years.",
        
        "imc2_para": "In-memory computing (IMC) is a technology that stores data in the main memory (RAM) of a computing system rather than on traditional disk storage. This approach allows for significantly faster data retrieval and processing times, making it ideal for applications requiring real-time or near-real-time data analysis and decision-making.",
        
        "imc_word": "In-memory computing",
        "imc2_word": "Processing in memory",
        
        # Pizza prompts
        "pizzas": "Pizza is a dish of Italian origin consisting of a usually round, flat base of leavened wheat-based dough topped with tomatoes, cheese, and often various other ingredients (such as anchovies, mushrooms, olives, vegetables, meat, etc.), baked at a high temperature, traditionally in a wood-fired oven.",
        
        "pizzas_para": "Neapolitan pizza is a style of pizza from Naples, Italy, known for its soft, thin dough with a raised, airy crust called the cornicione. It is traditionally cooked at very high temperatures in a wood-fired oven, which creates charred leoparding on the crust. Key characteristics include a simple topping of fresh, high-quality ingredients, such as tomatoes, fresh mozzarella, basil, and olive oil. Neapolitan pizza (Italian: pizza napoletana; Neapolitan: pizza napulitana) is the version of the round pizza typically prepared in the Italian city of Naples and characterised by a soft, thin dough with high edges.[1] The tomatoes are traditionally either San Marzano tomatoes or pomodorini del Piennolo del Vesuvio, which grow on the volcanic plains to the south of Mount Vesuvius, and the cheese is traditionally mozzarella di bufala campana or fior di latte di Agerola. Pizza napoletana is a traditional speciality guaranteed (TSG) product in the European Union and the United Kingdom, and the art of its making (arte del pizzaiolo napoletano) is included on UNESCO's list of intangible cultural heritage.",
        
        "pizzas_word": "Neapolitan pizza",
        
        # Actress prompts
        "actress": "Anne Jacqueline Hathaway is an American actress. Her accolades include an Academy Award, a British Academy Film Award, a Golden Globe Award, and a Primetime Emmy Award. Her films have grossed over $6.8 billion worldwide, and she appeared on the Forbes Celebrity 100 list in 2009.",
        
        "actress_para": "Anne Jacqueline Hathaway (born November 12, 1982) is an American actress. Her accolades include an Academy Award, a British Academy Film Award, a Golden Globe Award, and a Primetime Emmy Award. Her films have grossed over $6.8 billion worldwide, and she appeared on the Forbes Celebrity 100 list in 2009. She was among the world's highest-paid actresses in 2015. Hathaway performed in several plays in high school. As a teenager, she was cast in the television series Get Real (1999–2000) and made her breakthrough by playing the lead role in the Disney comedy The Princess Diaries (2001). After starring in a string of family films, including Ella Enchanted (2004), Hathaway made a transition to mature roles with the 2005 drama Brokeback Mountain. The comedy-drama The Devil Wears Prada (2006), in which she played an assistant to a fashion magazine editor, was her biggest commercial success to that point. She played a recovering addict in the drama Rachel Getting Married (2008), which earned her a nomination for the Academy Award for Best Actress.",
        
        "actress_word": "Anne Hathaway",
        
        # Other prompts
        "astrophysics": "Dark matter constitutes approximately 27 percent of the universe's mass-energy content, yet its composition remains one of the most profound mysteries in astrophysics. Recent studies suggest that",
        
        "astro_word": "Dark matter",
        
        "maths": "(2*3 + 5-1)/2 + 10*(4+2)-3=48",
    }
    
    # Get the base prompt text
    if prompt_type == "mmlu":
        if prompt_subject is None:
            raise ValueError("prompt_subject must be specified when prompt_type='mmlu'")
        prompt_text = get_mmlu_prompt_concat(prompt_subject)   # We don't pass max_samples because that picks the Question, Choices and Answer sets together.
    elif prompt_type == "custom":
        if prompt_subject is not None:
            if prompt_subject not in custom_prompts:
                available = ", ".join(custom_prompts.keys())
                raise ValueError(f"Unknown prompt_subjects '{prompt_subject}'. Available options: {available}")
            prompt_text = custom_prompts[prompt_subject]
        elif custom_text is not None:
            prompt_text = custom_text
        else:
            raise ValueError("Either prompt_subject or custom_text must be specified when prompt_type='custom'")
        
    # Tokenize the prompt
    tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
    
    # Validate against model's max sequence length
    if len(tokens) > max_seq_len:
        print(f"  WARNING: Prompt has {len(tokens)} tokens, which exceeds model's max sequence length of {max_seq_len}")
        print(f"  Truncating prompt to {max_seq_len} tokens")
        tokens = tokens[:max_seq_len]
        prompt_text = tokenizer.decode(tokens)
    
    # Apply user-specified prompt_length constraint (if provided)
    if prompt_length is not None:
        # First, validate that prompt_length doesn't exceed max_seq_len
        if prompt_length > max_seq_len:
            print(f"  WARNING: Requested prompt_length ({prompt_length}) exceeds model's max sequence length ({max_seq_len})")
            print(f"  Using max_seq_len ({max_seq_len}) instead")
            prompt_length = max_seq_len
        
        # Now apply the (validated) prompt_length
        if len(tokens) > prompt_length:
            print(f"  Prompt has {len(tokens)} tokens, truncating to {prompt_length} tokens")
            tokens = tokens[:prompt_length]
            prompt_text = tokenizer.decode(tokens)
        elif len(tokens) < prompt_length:
            print(f"  Prompt has {len(tokens)} tokens (less than requested {prompt_length})")
    
    # Convert to tensor
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    
    print(f"\nPrompt initialized:")
    print(f"  Type: {prompt_type}")
    print(f"  Subject: {prompt_subject}")
    print(f"  Length: {len(tokens)} tokens")
    print(f"  Model max seq len: {max_seq_len} tokens")
    
    return prompt_text, input_ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="gpt2", help='LLM model name or path')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--save_activations', action='store_true', help='Save activations during model run')
    parser.add_argument('--save_res_dir', type=str, default=None, help='Path to save activations and plots')
    parser.add_argument('--save_model', type=str, default="llm_weights", help='Path to save the model')
    
    # Prompt configuration arguments
    parser.add_argument('--prompt_type', type=str, default="custom", 
                       help='Type of prompt: mmlu, custom')
    parser.add_argument('--prompt_length', type=int, default=None,
                       help='Maximum prompt length in tokens. If None, uses full prompt.')
    parser.add_argument('--prompt_subject', type=str, default="imc",
                       help='Prompt subject name: Custom :: imc, imc2, imc3, imc_para, imc2_para, imc_word, '
                            'pizzas, pizzas_para, pizzas_word, actress, actress_para, actress_word, '
                            'astrophysics, astro_word, maths' \
                            ' | MMLU :: college_computer_science, machine_learning, electrical_engineering, business_ethics, world_religions, prehistory, moral_disputes')
    parser.add_argument('--custom_prompt_text', type=str, default=None,
                       help='Custom prompt text (required if prompt_type=custom and prompt_subject is not specified)')
    
    # Generation and pruning arguments
    parser.add_argument('--generation', type=int, default=0, help='Number of generations')
    parser.add_argument('--layer_topk', type=str, default=None, help='Layer-wise topk specification. Formats: '
                            '"all:0.6" (60%% to all), '
                            '"0,1,2:0.6" (60%% to layers 0,1,2), '
                            '"0:1000,1:1500" (specific counts), '
                            '"0-5:0.6" (60%% to layers 0-5), '
                            '"0-5:0.6,6-11:0.8" (mixed ranges)')
    parser.add_argument('--maskingStep', type=int, default=None, help='Step at which to start masking neurons')
    
    # Evaluation arguments
    #### Perplexity
    parser.add_argument('--eval_perplexity', action='store_true', help='Evaluate perplexity on datasets')
    parser.add_argument('--ppl_datasets', nargs='+', default=["custom","mmlu"], help='Datasets for perplexity eval. Options: custom(imc,anne_corpus,food_corpus), mmlu')
    parser.add_argument('--ppl_max_samples', type=int, default=None, help='Max samples for perplexity eval')
    parser.add_argument('--ppl_subjects', nargs='+', default=["imc"], help='Subjects for perplexity eval. Options: imc,anne_corpus,food_corpus,mmlu_subjects')
    #### MMLU
    parser.add_argument('--eval_mmlu', action='store_true', help='Evaluate MMLU benchmark')
    parser.add_argument('--mmlu_datasets', nargs='+', default=["college_computer_science", "machine_learning", "electrical_engineering", "business_ethics", "world_religions", "prehistory", "moral_disputes"], help='Datasets for mmlu eval')
    parser.add_argument('--mmlu_shots', type=int, default=0, help='Number of shots for MMLU evaluation')
    parser.add_argument('--mmlu_max_samples', type=int, default=None, help='Max samples for general mmlu eval')
    #### General NLP
    parser.add_argument('--eval_general_nlp', action='store_true', help='Evaluate general nlp benchmark')
    parser.add_argument('--general_nlp_datasets', nargs='+', default=None, help='Datasets for general nlp eval. Options: boolq, rte, hellaswag, winogrande, arc_easy, arc_challenge, openbookqa')
    parser.add_argument('--general_nlp_max_samples', type=int, default=None, help='Max samples for general nlp eval')
    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    times = {
        'start': 0.0,
        'model_load': 0.0,
        'prefill': 0.0,
        'generation': 0.0,
        'eval_ppl': 0.0,
        'eval_mmlu': 0.0,
        'eval_general_nlp': 0.0,
        'end': 0.0
    }

    print(f"Loading model: {args.model}")
    times['start'] = time.time()
    model = get_llm(args.model, args.cache_dir)
    times['model_load'] = time.time() - times['start']
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    # Detect model type
    model_type = detect_model_type(model)
    print(f"Detected model type: {model_type.upper()}")

    # Select device
    device = torch.device("cuda:0")
    if hasattr(model, "hf_device_map"):
        if model_type == 'gpt2':
            device = model.hf_device_map.get("lm_head", device)
        else:  # llama
            device = model.hf_device_map.get("lm_head", device)
    print("Using device:", device)

    # Datastructures for storing the activations.
    pre_ln1_activations = defaultdict(list)     # For storing the activations before the LayerNorm1 layers | This is for storing the residual
    pre_attn_activations = defaultdict(list)    # For storing the activations before the attention layers | This is also the output after layer norm 1.
    post_attn_activations = defaultdict(list)   # For storing the activations after the attention layers
    pre_ln2_activations = defaultdict(list)     # For storing the activations before the LayerNorm2 layers | This is post_attn_act + residual(pre_ln1)
    pre_mlp1_activations = defaultdict(list)     # For storing the activations before the MLP layers | This is equivalent to output of ln2.
    pre_mlp2_activations = defaultdict(list)     # For storing the activations of the neurons(inside the MLPs) | This is the neurons
    post_mlp2_activations = defaultdict(list)    # For storing the activations after the MLP layers
    post_layer_activations = defaultdict(list)  # For storing the activations after the complete transformer block
    mlp2_forward_proxy = {}

    # Get embedding weights based on model type
    if model_type == 'gpt2':
        embedding_weights = model.transformer.wte.weight.data
        num_layers = model.config.n_layer if model.config.n_layer is not None else 0
        intermediate_size = model.config.n_inner if model.config.n_inner is not None else 4 * model.config.hidden_size
    else:  # llama
        num_layers = model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else len(model.model.layers)
        intermediate_size = model.config.intermediate_size
        embedding_weights = model.model.embed_tokens.weight.data

    if args.layer_topk is not None:
        # Get model dimensions
        per_layer_config = parse_layer_topk(args.layer_topk, num_layers, intermediate_size)
        
        for layer, topk in sorted(per_layer_config.items()):
            pct = (topk / intermediate_size) * 100
            print(f"  Layer {layer}: {topk} neurons ({pct:.1f}%)")
    else:
        # No pruning
        per_layer_config = {}
        print("No pruning configuration specified")

    # Initialize NeuronDefuser with per-layer configuration
    neuronDefuser = NeuronDefuser(
        maskingStep=args.maskingStep, 
        per_layer_topk=per_layer_config,
        device=device
    )
    
    # Setup hooks based on model type
    if model_type == 'gpt2':
        hooks = setup_hooks_gpt2(
            model, neuronDefuser, pre_ln1_activations, pre_attn_activations,
            post_attn_activations, pre_ln2_activations, pre_mlp1_activations,
            pre_mlp2_activations, post_mlp2_activations, post_layer_activations,
            mlp2_forward_proxy, embedding_weights
        )
    else:  # llama
        hooks = setup_hooks_llama(
            model, neuronDefuser, pre_ln1_activations, pre_attn_activations,
            post_attn_activations, pre_ln2_activations, pre_mlp1_activations,
            pre_mlp2_activations, post_mlp2_activations, post_layer_activations,
            mlp2_forward_proxy, embedding_weights
        )

    # Tokenize and run prompt
    prompt_text, input_ids = initialize_prompt(
        tokenizer=tokenizer,
        device=device,
        max_seq_len=model.config.max_position_embeddings,
        prompt_type=args.prompt_type,
        prompt_length=args.prompt_length,
        prompt_subject=args.prompt_subject,
        custom_text=args.custom_prompt_text
    )

    # Store initial token count for later
    initial_token_count = input_ids.shape[1]

    # Prefill Phase
    start = time.time()
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    times['prefill'] = time.time() - start

    # Decode / Generation Phase
    start = time.time()
    past_key_values = outputs.past_key_values
    all_tokens = input_ids
    
    for i in range(args.generation):
        iter_start = time.time()
        with torch.no_grad():
            outputs = model(
                all_tokens[:, -1:],
                past_key_values=past_key_values,
                use_cache=True
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            all_tokens = torch.cat([all_tokens, next_token], dim=-1)
            past_key_values = outputs.past_key_values
        
        if i % 10 == 0:
            print(f"Token {i}: {time.time() - iter_start:.3f}s")
    
    times['generation'] = time.time() - start

    if args.save_activations:
        os.makedirs(args.save_res_dir, exist_ok=True)
        # Create a pickle file to save these datastructures.
        save_path = os.path.join(args.save_res_dir, 'activations.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump({
                'pre_ln1_activations': pre_ln1_activations,
                'pre_attn_activations': pre_attn_activations,
                'post_attn_activations': post_attn_activations,
                'pre_ln2_activations': pre_ln2_activations,
                'pre_mlp1_activations': pre_mlp1_activations,
                'pre_mlp2_activations': pre_mlp2_activations,
                'post_mlp2_activations': post_mlp2_activations,
                'post_layer_activations': post_layer_activations,
                'mlp2_forward_proxy': mlp2_forward_proxy,
                'model_type': model_type
            }, f)
        print(f"Activations saved to {save_path}")

    # Perf analysis
    if args.eval_perplexity:
        start_time = time.time()
        evaluate_perplexity(model, tokenizer, device, args.ppl_datasets, args.ppl_subjects, max_samples=args.ppl_max_samples, save_dir=args.save_res_dir)
        times['eval_ppl'] = time.time() - start_time
    if args.eval_mmlu:
        start_time = time.time()
        evaluate_mmlu_multi_shot(model, tokenizer, device, args.mmlu_datasets, max_samples=args.mmlu_max_samples, shots=args.mmlu_shots, save_dir=args.save_res_dir)
        times['eval_mmlu'] = time.time() - start_time
    if args.eval_general_nlp:
        start_time = time.time()
        evaluate_general_datasets(model, tokenizer, device, max_samples=args.general_nlp_max_samples, datasets=args.general_nlp_datasets, save_dir=args.save_res_dir)
        times['eval_general_nlp'] = time.time() - start_time
    
    # Remove hooks
    for hook in hooks:
        hook.remove()

    times['end'] = time.time()
    # Printing the input and output statements.
    print("\n" + "="*80)
    print("Generation Results:")
    print("="*80)
    print(f"Prompt type: {args.prompt_type}")
    print(f"Subject: {args.prompt_subject if args.prompt_subject is not None else None}")
    print(f"Initial prompt tokens: {initial_token_count}")
    print(f"Generated tokens: {args.generation}")
    print(f"Total tokens: {all_tokens.shape[1]}")
    #print("-"*80)
    #print(f"Initial prompt:\n{prompt_text[:500]}{'...' if len(prompt_text) > 500 else ''}")
    #print("-"*80)
    #print(f"Complete output:\n{tokenizer.decode(all_tokens[0])}")
    #print("-"*80)
    if args.generation > 0:
        new_tokens = all_tokens[0][initial_token_count:]
        print(f"Newly generated tokens: {new_tokens.tolist()}")
        print(f"Newly generated text:\n{tokenizer.decode(new_tokens)}")
    print("="*80 + "\n")

    print("\n" + "="*80)
    print("TIMING SUMMARY")
    print("="*80)
    print(f"TIMING_START={times['start']:.3f}")
    print(f"TIMING_MODEL_LOAD={times['model_load']:.3f}")
    print(f"TIMING_PREFILL={times['prefill']:.3f}")
    print(f"TIMING_GENERATION={times['generation']:.3f}")
    print(f"TIMING_EVAL_PPL={times['eval_ppl']:.3f}")
    print(f"TIMING_EVAL_MMLU={times['eval_mmlu']:.3f}")
    print(f"TIMING_EVAL_GENERAL_NLP={times['eval_general_nlp']:.3f}")
    print(f"TIMING_END={times['end']:.3f}")
    print(f"TIMING_TOTAL={times['end'] - times['start']:.3f}")
    print("="*80 + "\n")

    if args.maskingStep is not None:
        if args.save_res_dir is not None:
            os.makedirs(args.save_res_dir, exist_ok=True)
        neuronDefuser.save_results(output_dir=args.save_res_dir)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

def evaluate_perplexity(model, tokenizer, device, perplexity_datasets, perplexity_subjects, max_samples=None, save_dir=None):
    """Evaluate model perplexity on domain-specific datasets."""
    
    datasets_to_eval = []

    for dataset in perplexity_datasets:
        if dataset == "custom":
            for subject in perplexity_subjects:
                if subject == "in_memory_computing" or subject == "imc":
                    datasets_to_eval.append((dataset, subject, os.path.join(DATASETS_DIR, "in_memory_computing", "in_memory_computing_dataset")))
                elif subject == "food_corpus":
                    datasets_to_eval.append((dataset, subject, os.path.join(DATASETS_DIR, "food_corpus", "food_corpus_dataset")))
                elif subject == "anne_corpus":
                    datasets_to_eval.append((dataset, subject, os.path.join(DATASETS_DIR, "anne_corpus", "anne_corpus_dataset")))
        elif dataset == "mmlu":
            # Extract the actual subject name (e.g., "mmlu_college_computer_science" -> "college_computer_science")
            for subject in perplexity_subjects:      
                datasets_to_eval.append((dataset, subject, ""))
                print(f"Added MMLU dataset: {subject}")
        else:
            print(f"Warning: Unknown subject '{subject}', skipping...")
    
    if not datasets_to_eval:
        print("No valid datasets to evaluate!")
        return {}
    
    results = evaluate_on_datasets(
        model, tokenizer,
        datasets=datasets_to_eval,
        device=device,
        max_samples=max_samples,
        max_length_ppl=1024
    )
        
    # Save perplexity results to a dedicated file
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        ppl_output_path = os.path.join(save_dir, "perplexity_results.json")
        with open(ppl_output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Perplexity results saved to: {ppl_output_path}")

def evaluate_mmlu_multi_shot(model, tokenizer, device, subjects, max_samples=None, shots=0, save_dir=None):
    """Evaluate few-shot learning capabilities."""
    if not LM_EVAL_AVAILABLE:
        print("Error: lm-evaluation-harness not installed. Skipping MMLU evaluation.")
        return
    
    if save_dir:
        output_path = os.path.join(save_dir, f"results_mmlu_{shots}_shots.json")
    else:
        output_path = os.path.join(RESULTS_DIR, "mmlu", f"results_mmlu_{shots}_shots.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    task_names = [f"mmlu_{subject}" for subject in subjects]
    
    print(f"\nEvaluating MMLU with {shots} shots")
    print(f"Tasks: {task_names}")
    print(f"Max samples per task: {max_samples if max_samples else 'All'}")

    wrapped_model = HFLM(
        pretrained=model,      # Your edited model with hooks
        tokenizer=tokenizer,   # Your tokenizer
        batch_size=32,          # Keep at 1 for safety with hooks
        device=str(device)
    )

    results = simple_evaluate(
        model=wrapped_model,  # Your model with NeuronDefuser hooks
        model_args=None,  # Not needed since we're passing the model directly
        tasks=task_names,
        num_fewshot=shots,
        batch_size=32,  # Keep at 1 to avoid issues with your hooks
        device=str(device),
        limit=max_samples,  # Limit samples per task (None = all samples)
        log_samples=True,  # Set True if you want detailed sample-level results
    )

    # Save results with proper serialization
    with open(output_path, 'w') as f:
        json.dump(
            results, 
            f, 
            indent=2,
            default=handle_non_serializable,  # ✓ This handles torch.dtype and other non-serializable objects
            ensure_ascii=False
        )
    
    print(f"\nEvaluation complete!")
    print(f"Results saved to: {output_path}")
    
    # Print summary
    if 'results' in results:
        print("\n" + "="*80)
        print("MMLU Evaluation Results:")
        print("="*80)
        for task, metrics in results['results'].items():
            acc = metrics.get('acc,none', metrics.get('acc', 'N/A'))
            if isinstance(acc, float):
                print(f"{task:50s}: {acc*100:.2f}%")
            else:
                print(f"{task:50s}: {acc}")
        print("="*80 + "\n")
    
    return results

def evaluate_general_datasets(model, tokenizer, device, max_samples=None, datasets=None, save_dir=None):
    """Evaluate general NLP datasets."""
    if save_dir:
        output_path = os.path.join(save_dir, f"results_general_nlp.json")
    else:
        output_path = os.path.join(RESULTS_DIR, "general_nlp", f"results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results = evaluate_general_nlp(model, tokenizer, device, max_samples=max_samples, datasets=datasets)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")

if __name__ == '__main__':
    main()