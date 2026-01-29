#!/bin/bash

#############################################################################
# Experiment Wrapper Script for Dynamic Pruning
# Usage: ./run_experiment.sh
# 
# Features:
# - Logs all output to files
# - Tracks start/end times
# - Saves experiment configuration
# - Creates organized result directories
#############################################################################

# ============================================================================
# CONFIGURATION - Modify these parameters for different experiments
# ============================================================================

DEVICE=6                  # CUDA device ID

# Model configuration
MODE="auto"             # Options: "manual", "auto"
#MODEL="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
#MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
#MODEL="mistralai/Mistral-7B-Instruct-v0.3"
#MODEL="Qwen/Qwen3-14B-Base"
MODEL="meta-llama/Llama-3.2-3B" # Options: "gpt2", "gpt2-xl", "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.2-3B","Llama-2-7b-hf"  etc.
#MODEL="Qwen/Qwen3-4B-Base"
#MODEL="meta-llama/Llama-3.1-8B"
#MODEL="meta-llama/Llama-2-7b-hf"
#MODEL="meta-llama/Llama-2-13b-hf"
CACHE_DIR="llm_weights"
SEED=0                    # Random seed for reproducibility
SAVE_MODEL=""             # Path to save the model (leave empty to skip)

# Prompt configuration
PROMPT_TYPE="mmlu"           # Options: "custom", "mmlu"
PROMPT_SUBJECT="marketing"           # For custom: "imc", "pizzas", "actress", etc.
                               # For mmlu: "college_computer_science", etc.
PROMPT_LENGTH=4000               # Leave empty for None (no prompt length limit)
CUSTOM_PROMPT_TEXT=""        # Custom prompt text (leave empty to use prompt_subject)

# Pruning configuration
## Uncomment and set these for specific pruning, comment the set below
#LAYER_TOPK="all:auto"
#LAYER_TOPK_3_2_3B="0:89.36,1:73.14,2:71.52,3:69.10,4:88.04,5:90.00,6:89.36,7:87.90,8:87.49,9:82.10,10:79.28,11:58.62,12:59.11,13:53.46,14:43.72,15:15.79,16:10.00,17:10.00,18:10.00,19:10.00,20:10.00,21:10.00,22:10.00,23:10.00,24:10.00,25:28.09,26:53.96,27:89.96"              # Layer-specific top-k configuration (leave empty for None)

LAYER_TOPK_3_2_3B_72="0:0.8936,1:0.914,2:0.952,3:0.910,4:0.904,5:0.9000,6:0.8936,7:0.8790,8:0.8749,9:0.910,10:0.9928,11:0.9862,12:0.9911,13:0.9346,14:0.9372,15:0.9579,16:0.9000,17:0.9000,18:0.9000,19:0.9000,20:0.9000,21:0.9000,22:0.9000,23:0.9000,24:0.9000,25:0.909,26:0.896,27:0.8996"              # Layer-specific top-k configuration (leave empty for None)

LAYER_TOPK_3_2_3B="0:0.8936,1:0.7314,2:0.7152,3:0.6910,4:0.8804,5:0.9000,6:0.8936,7:0.8790,8:0.8749,9:0.8210,10:0.8928,11:0.8862,12:0.8911,13:0.8346,14:0.8372,15:0.8579,16:0.1000,17:0.1000,18:0.1000,19:0.1000,20:0.1000,21:0.1000,22:0.1000,23:0.1000,24:0.1000,25:0.289,26:0.536,27:0.8996"              # Layer-specific top-k configuration (leave empty for None)
LAYER_TOPK_3_2_3B_NO_MAG="0:0.7172,1:0.6579,2:0.6148,3:0.5903,4:0.8746,5:0.90,6:0.899,7:0.89,8:0.8981,9:0.8871,10:0.8766,11:0.6749,12:0.6715,13:0.6192,14:0.468,15:0.1254,16:0.10,17:0.10,18:0.10,19:0.10,20:0.10,21:0.10,22:0.10,23:0.10,24:0.10,25:0.31,26:0.59,27:0.8325"
LAYER_TOPK_3_2_3B_NO_MAG_NO_BORDER="0:0.25,1:0.338,2:0.3133,3:0.3854,4:0.8916,5:0.90,6:0.8998,7:0.8983,8:0.8997,9:0.898,10:0.8964,11:0.7845,12:0.7822,13:0.7475,14:0.8473,15:0.8202,16:0.3561,17:0.2485,18:0.171,19:0.10,20:0.10,21:0.10,22:0.10,23:0.10,24:0.10,25:0.3192,26:0.4703,27:0.6827"
LAYER_TOPK_3_2_3B_NO_BORDER="0:0.8933,1:0.4035,2:0.5019,3:0.5544,4:0.8745,5:0.90,6:0.8979,7:0.8939,8:0.8937,9:0.8638,10:0.8424,11:0.6851,12:0.6889,13:0.6459,14:0.5717,15:0.3592,16:0.2585,17:0.1664,18:0.1071,19:0.10,20:0.10,21:0.10,22:0.10,23:0.10,24:0.10,25:0.1845,26:0.3137,27:0.8997"
LAYER_TOPK_3_2_3B_70="0:0.8393,1:0.2061,2:0.1579,3:0.1,4:0.6976,5:0.9,6:0.8389,7:0.6754,8:0.6302,9:0.4709,10:0.3876,11:0.1,12:0.1,13:0.1,14:0.1,15:0.1,16:0.1,17:0.1,18:0.1,19:0.1,20:0.1,21:0.1,22:0.1,23:0.1,24:0.1,25:0.1,26:0.1,27:0.896"
LAYER_TOPK_3_2_3B_70_NO_BORDER="0:0.695,1:0.1,2:0.1,3:0.1,4:0.622,5:0.9,6:0.8461,7:0.7198,8:0.7105,9:0.5897,10:0.5251,11:0.1,12:0.1,13:0.1,14:0.1,15:0.1,16:0.1,17:0.1,18:0.1,19:0.1,20:0.1,21:0.1,22:0.1,23:0.1,24:0.1,25:0.1,26:0.1,27:0.8919"

LAYER_TOPK_3_1_8B="0:0.8864,1:0.7743,2:0.7547,3:0.7199,4:0.7534,5:0.8033,6:0.7902,7:0.7423,8:0.7213,9:0.6537,10:0.648,11:0.6309,12:0.6044,13:0.5859,14:0.5832,15:0.5181,16:0.4282,17:0.3293,18:0.2601,19:0.2263,20:0.1817,21:0.1186,22:0.1015,23:0.101,24:0.101,25:0.1095,26:0.1104,27:0.2221,28:0.3671,29:0.5515,30:0.7218,31:0.9"
LAYER_TOPK_3_1_8B_NO_BORDER="0:0.6137,1:0.4198,2:0.493,3:0.5161,4:0.6339,5:0.744,6:0.7571,7:0.7311,8:0.7354,9:0.6984,10:0.7008,11:0.6863,12:0.6638,13:0.648,14:0.6457,15:0.5904,16:0.514,17:0.43,18:0.3711,19:0.3424,20:0.3045,21:0.2508,22:0.2364,23:0.1876,24:0.2134,25:0.2431,26:0.2439,27:0.2728,28:0.3009,29:0.3946,30:0.5172,31:0.9"
LAYER_TOPK_3_1_8B_70="0:0.7828,1:0.5686,2:0.5311,3:0.4646,4:0.5286,5:0.624,6:0.5991,7:0.5074,8:0.4673,9:0.3381,10:0.3273,11:0.2945,12:0.2439,13:0.2084,14:0.2033,15:0.1,16:0.1,17:0.1,18:0.1,19:0.1,20:0.1,21:0.1,22:0.1,23:0.1,24:0.1,25:0.1,26:0.1,27:0.1,28:0.1,29:0.1428,30:0.4682,31:0.9"
LAYER_TOPK_3_1_8B_70_NO_BORDER="0:0.3868,1:0.1,2:0.1953,3:0.2319,4:0.4189,5:0.5937,6:0.6144,7:0.5731,8:0.58,9:0.5213,10:0.5252,11:0.5021,12:0.4663,13:0.4413,14:0.4377,15:0.3499,16:0.2286,17:0.1,18:0.1,19:0.1,20:0.1,21:0.1,22:0.1,23:0.1,24:0.1,25:0.1,26:0.1,27:0.1,28:0.1,29:0.1,30:0.2336,31:0.9"

LAYER_TOPK_2_7b="0:0.6344,1:0.7179,2:0.6798,3:0.7629,4:0.7294,5:0.8002,6:0.8124,7:0.8171,8:0.7799,9:0.7257,10:0.7329,11:0.7176,12:0.7215,13:0.6268,14:0.5540,15:0.4764,16:0.3374,17:0.2967,18:0.2265,19:0.1587,20:0.1238,21:0.1361,22:0.1164,23:0.1445,24:0.1156,25:0.2226,26:0.1455,27:0.2758,28:0.3781,29:0.4487,30:0.6846,31:0.9000"              # Layer-specific top-k configuration (leave empty for None)
LAYER_TOPK_2_7b_NO_MAG_NO_BORDER="0:0.1,1:0.1,2:0.1972,3:0.6299,4:0.596,5:0.8744,6:0.8943,7:0.8975,8:0.8969,9:0.8953,10:0.8985,11:0.8988,12:0.9,13:0.872,14:0.7738,15:0.6629,16:0.4479,17:0.4053,18:0.3078,19:0.2072,20:0.1507,21:0.1874,22:0.1604,23:0.2142,24:0.1644,25:0.3622,26:0.224,27:0.3307,28:0.3003,29:0.1756,30:0.5308,31:0.7436"
LAYER_TOPK_2_7b_70="0:0.2855,1:0.4487,2:0.3741,3:0.5367,4:0.4712,5:0.6095,6:0.6333,7:0.6425,8:0.5698,9:0.464,10:0.4781,11:0.4482,12:0.4557,13:0.2707,14:0.1284,15:0.1,16:0.1,17:0.1,18:0.1,19:0.1,20:0.1,21:0.1,22:0.1,23:0.1,24:0.1,25:0.1,26:0.1,27:0.1,28:0.1,29:0.1,30:0.3837,31:0.9"
LAYER_TOPK_2_7b_70_NO_MAG_NO_BORDER="0:0.1,1:0.1,2:0.1,3:0.1,4:0.1,5:0.6157,6:0.7189,7:0.8283,8:0.8124,9:0.768,10:0.8587,11:0.8667,12:0.9,13:0.6082,14:0.3078,15:0.1,16:0.1,17:0.1,18:0.1,19:0.1,20:0.1,21:0.1,22:0.1,23:0.1,24:0.1,25:0.1,26:0.1,27:0.1,28:0.1,29:0.1,30:0.1,31:0.2155"
LAYER_TOPK_2_7b_70_NO_BORDER="0:0.1,1:0.1,2:0.1,3:0.3363,4:0.3491,5:0.5787,6:0.6479,7:0.6907,8:0.6613,9:0.6129,10:0.6321,11:0.6111,12:0.6164,13:0.486,14:0.3857,15:0.2788,16:0.1,17:0.1,18:0.1,19:0.1,20:0.1,21:0.1,22:0.1,23:0.1,24:0.1,25:0.1,26:0.1,27:0.1,28:0.1,29:0.1,30:0.1132,31:0.9"
LAYER_TOPK_2_7b_80="0:0.1,1:0.2094,2:0.1024,3:0.3355,4:0.2416,5:0.44,6:0.4741,7:0.4873,8:0.383,9:0.2312,10:0.2514,11:0.2086,12:0.2194,13:0.1,14:0.1,15:0.1,16:0.1,17:0.1,18:0.1,19:0.1,20:0.1,21:0.1,22:0.1,23:0.1,24:0.1,25:0.1,26:0.1,27:0.1,28:0.1,29:0.1,30:0.1161,31:0.9"
LAYER_TOPK_2_7b_85="0:0.1,1:0.1,2:0.1,3:0.1464,4:0.1,5:0.2806,6:0.3244,7:0.3413,8:0.2074,9:0.1,10:0.1,11:0.1,12:0.1,13:0.1,14:0.1,15:0.1,16:0.1,17:0.1,18:0.1,19:0.1,20:0.1,21:0.1,22:0.1,23:0.1,24:0.1,25:0.1,26:0.1,27:0.1,28:0.1,29:0.1,30:0.1,31:0.9"

LAYER_TOPK_2_13b="0:63.29,1:59.93,2:62.05,3:76.61,4:78.43,5:81.69,6:84.46,7:90.00,8:90.00,9:90.00,10:90.00,11:89.40,12:88.03,13:82.68,14:76.32,15:71.49,16:69.44,17:58.85,18:46.01,19:33.96,20:28.94,21:19.94,22:10.50,23:10.66,24:12.47,25:10.51,26:10.51,27:13.84,28:10.50,29:14.63,30:10.51,31:14.72,32:10.50,33:16.48,34:20.36,35:38.82,36:52.26,37:54.72,38:69.89,39:86.61"
LAYER_TOPK_2_13b_70="0:0.1353,1:0.1,2:0.1062,3:0.449,4:0.492,5:0.5686,6:0.634,7:0.9,8:0.7858,9:0.7872,10:0.8553,11:0.7503,12:0.7179,13:0.5921,14:0.4422,15:0.3285,16:0.2802,17:0.1,18:0.1,19:0.1,20:0.1,21:0.1,22:0.1,23:0.1,24:0.1,25:0.1,26:0.1,27:0.1,28:0.1,29:0.1,30:0.1,31:0.1,32:0.1,33:0.1,34:0.1,35:0.1,36:0.1,37:0.1,38:0.2907,39:0.6847"
LAYER_TOPK_2_13b_80="0:0.1,1:0.1,2:0.1,3:0.1,4:0.1,5:0.1881,6:0.3111,7:0.9,8:0.5968,9:0.5995,10:0.7667,11:0.53,12:0.4691,13:0.2322,14:0.1,15:0.1,16:0.1,17:0.1,18:0.1,19:0.1,20:0.1,21:0.1,22:0.1,23:0.1,24:0.1,25:0.1,26:0.1,27:0.1,28:0.1,29:0.1,30:0.1,31:0.1,32:0.1,33:0.1,34:0.1,35:0.1,36:0.1,37:0.1,38:0.1,39:0.4065"
LAYER_TOPK_2_13b_85="0:0.1,1:0.1,2:0.1,3:0.1,4:0.1,5:0.1,6:0.1,7:0.9,8:0.3432,9:0.3476,10:0.6398,11:0.2343,12:0.1351,13:0.1,14:0.1,15:0.1,16:0.1,17:0.1,18:0.1,19:0.1,20:0.1,21:0.1,22:0.1,23:0.1,24:0.1,25:0.1,26:0.1,27:0.1,28:0.1,29:0.1,30:0.1,31:0.1,32:0.1,33:0.1,34:0.1,35:0.1,36:0.1,37:0.1,38:0.1,39:0.1"

LAYER_TOPK_DEEPSEEK_QWEN3_70="0:0.7017,1:0.5165,2:0.5085,3:0.3992,4:0.2427,5:0.172,6:0.2818,7:0.4571,8:0.3655,9:0.5868,10:0.4687,11:0.3624,12:0.3476,13:0.4624,14:0.378,15:0.4312,16:0.4791,17:0.2629,18:0.2468,19:0.1016,20:0.1823,21:0.2023,22:0.1246,23:0.1,24:0.1,25:0.1,26:0.1,27:0.1,28:0.1,29:0.1,30:0.1,31:0.1,32:0.1825,33:0.2912,34:0.2445,35:0.9"
LAYER_TOPK_DEEPSEEK_LLAMA_70="0:0.5295,1:0.1,2:0.1,3:0.1,4:0.2558,5:0.7896,6:0.9,7:0.8185,8:0.8648,9:0.6537,10:0.8377,11:0.8219,12:0.1363,13:0.2353,14:0.1931,15:0.1,16:0.1,17:0.1,18:0.1,19:0.1,20:0.1,21:0.1,22:0.1,23:0.1,24:0.1,25:0.1,26:0.1,27:0.1,28:0.1,29:0.1,30:0.1,31:0.6637"
LAYER_TOPK_QWEN3_14B_70="0:0.7264,1:0.6214,2:0.5726,3:0.5269,4:0.44,5:0.4572,6:0.3954,7:0.3136,8:0.3136,9:0.1611,10:0.2202,11:0.2825,12:0.3107,13:0.3227,14:0.293,15:0.285,16:0.2829,17:0.301,18:0.3062,19:0.3898,20:0.2541,21:0.1939,22:0.2078,23:0.2414,24:0.1937,25:0.1501,26:0.1,27:0.1118,28:0.1,29:0.1,30:0.1,31:0.1,32:0.1,33:0.1,34:0.1,35:0.1976,36:0.3132,37:0.4182,38:0.496,39:0.9"
LAYER_TOPK_QWEN3_4B_70="0:0.7369,1:0.6401,2:0.5805,3:0.4915,4:0.3958,5:0.3476,6:0.3438,7:0.3631,8:0.338,9:0.4247,10:0.3269,11:0.2897,12:0.2717,13:0.328,14:0.2789,15:0.3058,16:0.3318,17:0.2321,18:0.2415,19:0.1678,20:0.1934,21:0.2002,22:0.1567,23:0.1,24:0.1,25:0.1,26:0.1,27:0.1,28:0.1,29:0.1,30:0.1,31:0.1,32:0.2188,33:0.3569,34:0.4377,35:0.9"
LAYER_TOPK_MISTRAL_70="0:0.6167,1:0.5258,2:0.6126,3:0.5308,4:0.4596,5:0.4432,6:0.47,7:0.4646,8:0.6287,9:0.3623,10:0.4812,11:0.3195,12:0.3847,13:0.2734,14:0.1603,15:0.1619,16:0.111,17:0.1,18:0.1,19:0.1,20:0.1,21:0.1,22:0.1,23:0.1,24:0.1,25:0.1,26:0.1,27:0.1,28:0.1,29:0.1379,30:0.356,31:0.9"
#LAYER_TOPK=$LAYER_TOPK_3_1_8B_70
LAYER_TOPK=$LAYER_TOPK_3_2_3B_70
#LAYER_TOPK=$LAYER_TOPK_2_7b_85
#LAYER_TOPK=$LAYER_TOPK_QWEN3_14B_70
#LAYER_TOPK="all:0.3"

MASKING_STEP=0
RELEASE_STEP=""              # Step at which to release masked neurons (leave empty for None)
GENERATION=0
EMA_DECAY=""
RANKING_METHOD="magnitude"
PRUNE_STRATEGY="topk"
TOTAL_PRUNE_PERCENT=70.0
VERBOSE=true                 # Enable verbose output from NeuronDefuser

## Comment these out when pruning.
# LAYER_TOPK=""
# MASKING_STEP=""
# GENERATION=0
# EMA_DECAY="0.5"                    # Decay factor for EMA (0.0 to 1.0, leave empty for None/L2 norm)
# RANKING_METHOD="max"        # Method to rank neurons - max, mean, combined, product, magnitude
# PRUNE_STRATEGY="auto"           # Pruning strategy - topk, auto

# Activation saving configuration
SAVE_ACTIVATIONS=false       # Set to true to save activations (uses more memory)

# Evaluation configuration - Perplexity
EVAL_PERPLEXITY=false           # Set to true to enable perplexity evaluation
PPL_DATASETS="mmlu"          # Options: "custom", "mmlu" (space-separated)
PPL_SUBJECTS="college_computer_science abstract_algebra high_school_biology high_school_world_history marketing philosophy professional_law"             # Subjects for perplexity evaluation (space-separated)
#PPL_SUBJECTS="college_computer_science abstract_algebra high_school_biology virology high_school_world_history marketing philosophy professional_law world_religions business_ethics moral_disputes machine_learning"             # Subjects for perplexity evaluation (space-separated)
                               # For custom: "imc", "food_corpus", "anne_corpus"
                               # For mmlu: "college_computer_science", "machine_learning", etc.
PPL_MAX_SAMPLES=200             # Max samples for perplexity eval (leave empty for all)

# Evaluation configuration - MMLU
EVAL_MMLU=false             # Set to true to enable MMLU evaluation
MMLU_DATASETS="college_computer_science high_school_world_history marketing philosophy"
#MMLU_DATASETS="college_computer_science abstract_algebra high_school_biology virology high_school_world_history marketing philosophy professional_law world_religions business_ethics moral_disputes machine_learning"
                               # MMLU subjects to evaluate (space-separated)
MMLU_SHOTS=2                   # Number of few-shot examples (0=zero-shot, 5=five-shot)
MMLU_MAX_SAMPLES=200            # Max samples per MMLU task (leave empty for all)

# Evaluation configuration - General NLP
EVAL_GENERAL_NLP=true         # Set to true to enable general NLP evaluation
#GENERAL_NLP_DATASETS="mmlu gpqa medmcqa"        # Options: "boolq rte hellaswag winogrande arc_easy arc_challenge openbookqa"
GENERAL_NLP_DATASETS=""
GENERAL_NLP_MAX_SAMPLES=""     # Max samples for general NLP eval (leave empty for all)

# Results directory
EXPERIMENT_NAME=${TOTAL_PRUNE_PERCENT}prune_GRIFFIN_OURS
BASE_RESULTS_DIR="/users/grad/abhishektyagi/wanda/wanda/results/lm_eval_all/full"
MODEL_SAFE_NAME=$(echo "$MODEL" | sed 's/\//_/g')  # Replace / with _
RESULTS_DIR="${BASE_RESULTS_DIR}/${MODEL_SAFE_NAME}/${EXPERIMENT_NAME}"

# ============================================================================
# SETUP
# ============================================================================

# Create results directory
mkdir -p "$RESULTS_DIR"

# Generate timestamp
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
START_EPOCH=$(date +%s)

# Define log files - ONLY ONE TIMING LOG
OUTPUT_LOG="${RESULTS_DIR}/output_${TIMESTAMP}.log"
TIMING_LOG="${RESULTS_DIR}/timing_${TIMESTAMP}.log"  # Single timing file
CONFIG_LOG="${RESULTS_DIR}/config_${TIMESTAMP}.json"
LATEST_LINK="${RESULTS_DIR}/latest_run.log"
LATEST_TIMING="${RESULTS_DIR}/latest_timing.log"  # Symlink to latest timing

# ============================================================================
# PRINT CONFIGURATION
# ============================================================================

print_config() {
    echo "========================================" 
    echo "EXPERIMENT CONFIGURATION"
    echo "========================================" 
    echo "Experiment Name: $EXPERIMENT_NAME"
    echo "Mode: $MODE"
    echo "Model: $MODEL"
    echo "Seed: $SEED"
    echo "Prompt Type: $PROMPT_TYPE"
    echo "Prompt Subject: $PROMPT_SUBJECT"
    echo "Prompt Length: ${PROMPT_LENGTH:-None}"
    echo "Custom Prompt Text: ${CUSTOM_PROMPT_TEXT:-None}"
    echo "Layer TopK: ${LAYER_TOPK:-None}"
    echo "Masking Step: ${MASKING_STEP:-None}"
    echo "Release Step: ${RELEASE_STEP:-None}"
    echo "EMA Decay: ${EMA_DECAY:-None (L2 norm)}"
    echo "Ranking Method: $RANKING_METHOD"
    echo "Prune Strategy: $PRUNE_STRATEGY"
    echo "Total Prune Percent: $TOTAL_PRUNE_PERCENT%"
    echo "Generation Tokens: $GENERATION"
    echo "Verbose: $VERBOSE"
    echo "Save Activations: $SAVE_ACTIVATIONS"
    echo "Save Model: ${SAVE_MODEL:-None}"
    echo ""
    echo "Evaluation Settings:"
    echo "  Perplexity: $EVAL_PERPLEXITY"
    if [ "$EVAL_PERPLEXITY" = true ]; then
        echo "    - Datasets: $PPL_DATASETS"
        echo "    - Subjects: $PPL_SUBJECTS"
        echo "    - Max Samples: ${PPL_MAX_SAMPLES:-All}"
    fi
    echo "  MMLU: $EVAL_MMLU"
    if [ "$EVAL_MMLU" = true ]; then
        echo "    - Subjects: $MMLU_DATASETS"
        echo "    - Shots: $MMLU_SHOTS"
        echo "    - Max Samples: ${MMLU_MAX_SAMPLES:-All}"
    fi
    echo "  General NLP: $EVAL_GENERAL_NLP"
    if [ "$EVAL_GENERAL_NLP" = true ]; then
        echo "    - Datasets: ${GENERAL_NLP_DATASETS:-Default}"
        echo "    - Max Samples: ${GENERAL_NLP_MAX_SAMPLES:-All}"
    fi
    echo ""
    echo "Results Directory: $RESULTS_DIR"
    echo "Started at: $START_TIME"
    echo "========================================" 
}

# Print to terminal and timing log (single file)
print_config | tee "$TIMING_LOG"

# ============================================================================
# SAVE CONFIGURATION TO JSON
# ============================================================================

# Determine prompt_length value for JSON (null if empty)
if [ -z "$PROMPT_LENGTH" ]; then
    PROMPT_LENGTH_JSON="null"
else
    PROMPT_LENGTH_JSON="$PROMPT_LENGTH"
fi

# Determine max samples for JSON
if [ -z "$PPL_MAX_SAMPLES" ]; then
    PPL_MAX_SAMPLES_JSON="null"
else
    PPL_MAX_SAMPLES_JSON="$PPL_MAX_SAMPLES"
fi

if [ -z "$MMLU_MAX_SAMPLES" ]; then
    MMLU_MAX_SAMPLES_JSON="null"
else
    MMLU_MAX_SAMPLES_JSON="$MMLU_MAX_SAMPLES"
fi

if [ -z "$GENERAL_NLP_MAX_SAMPLES" ]; then
    GENERAL_NLP_MAX_SAMPLES_JSON="null"
else
    GENERAL_NLP_MAX_SAMPLES_JSON="$GENERAL_NLP_MAX_SAMPLES"
fi

# Determine values for JSON (null if empty)
if [ -z "$CUSTOM_PROMPT_TEXT" ]; then
    CUSTOM_PROMPT_TEXT_JSON="null"
else
    CUSTOM_PROMPT_TEXT_JSON="\"$CUSTOM_PROMPT_TEXT\""
fi

if [ -z "$LAYER_TOPK" ]; then
    LAYER_TOPK_JSON="null"
else
    LAYER_TOPK_JSON="\"$LAYER_TOPK\""
fi

if [ -z "$MASKING_STEP" ]; then
    MASKING_STEP_JSON="null"
else
    MASKING_STEP_JSON="$MASKING_STEP"
fi

if [ -z "$RELEASE_STEP" ]; then
    RELEASE_STEP_JSON="null"
else
    RELEASE_STEP_JSON="$RELEASE_STEP"
fi

if [ -z "$EMA_DECAY" ]; then
    EMA_DECAY_JSON="null"
else
    EMA_DECAY_JSON="$EMA_DECAY"
fi

if [ -z "$SAVE_MODEL" ]; then
    SAVE_MODEL_JSON="null"
else
    SAVE_MODEL_JSON="\"$SAVE_MODEL\""
fi

cat > "$CONFIG_LOG" << EOF
{
  "experiment_name": "$EXPERIMENT_NAME",
  "timestamp": "$TIMESTAMP",
  "start_time": "$START_TIME",
  "mode": "$MODE",
  "model": "$MODEL",
  "cache_dir": "$CACHE_DIR",
  "seed": $SEED,
  "save_activations": $SAVE_ACTIVATIONS,
  "save_model": $SAVE_MODEL_JSON,
  "prompt": {
    "type": "$PROMPT_TYPE",
    "subject": "$PROMPT_SUBJECT",
    "length": $PROMPT_LENGTH_JSON,
    "custom_text": $CUSTOM_PROMPT_TEXT_JSON
  },
  "pruning": {
    "layer_topk": $LAYER_TOPK_JSON,
    "masking_step": $MASKING_STEP_JSON,
    "release_step": $RELEASE_STEP_JSON,
    "ema_decay": $EMA_DECAY_JSON,
    "ranking_method": "$RANKING_METHOD",
    "prune_strategy": "$PRUNE_STRATEGY",
    "total_prune_percent": $TOTAL_PRUNE_PERCENT,
    "generation": $GENERATION,
    "verbose": $VERBOSE
  },
  "evaluation": {
    "perplexity": {
      "enabled": $EVAL_PERPLEXITY,
      "datasets": "$PPL_DATASETS",
      "subjects": "$PPL_SUBJECTS",
      "max_samples": $PPL_MAX_SAMPLES_JSON
    },
    "mmlu": {
      "enabled": $EVAL_MMLU,
      "datasets": "$MMLU_DATASETS",
      "shots": $MMLU_SHOTS,
      "max_samples": $MMLU_MAX_SAMPLES_JSON
    },
    "general_nlp": {
      "enabled": $EVAL_GENERAL_NLP,
      "datasets": "$GENERAL_NLP_DATASETS",
      "max_samples": $GENERAL_NLP_MAX_SAMPLES_JSON
    }
  },
  "results_dir": "$RESULTS_DIR"
}
EOF

echo "Configuration saved to: $CONFIG_LOG" | tee -a "$TIMING_LOG"

# ============================================================================
# BUILD COMMAND
# ============================================================================

CMD="CUDA_VISIBLE_DEVICES=$DEVICE python -u dynamicPrune.py \
    --mode \"$MODE\" \
    --model \"$MODEL\" \
    --seed $SEED \
    --cache_dir \"$CACHE_DIR\" \
    --prompt_type \"$PROMPT_TYPE\" \
    --prompt_subject \"$PROMPT_SUBJECT\""

# Add prompt_length only if it's set (not empty)
if [ -n "$PROMPT_LENGTH" ]; then
    CMD="$CMD \
    --prompt_length $PROMPT_LENGTH"
fi

# Add custom_prompt_text if set
if [ -n "$CUSTOM_PROMPT_TEXT" ]; then
    CMD="$CMD \
    --custom_prompt_text \"$CUSTOM_PROMPT_TEXT\""
fi

# Add save_activations flag if enabled
if [ "$SAVE_ACTIVATIONS" = true ]; then
    CMD="$CMD \
    --save_activations"
fi

# Add layer_topk only if set
if [ -n "$LAYER_TOPK" ]; then
    CMD="$CMD \
    --layer_topk \"$LAYER_TOPK\""
fi

# Add maskingStep only if set
if [ -n "$MASKING_STEP" ]; then
    CMD="$CMD \
    --maskingStep $MASKING_STEP"
fi

# Add releaseStep only if set
if [ -n "$RELEASE_STEP" ]; then
    CMD="$CMD \
    --releaseStep $RELEASE_STEP"
fi

# Add ema_decay only if set
if [ -n "$EMA_DECAY" ]; then
    CMD="$CMD \
    --ema_decay $EMA_DECAY"
fi

# Add ranking_method
CMD="$CMD \
    --ranking_method \"$RANKING_METHOD\""

# Add prune_strategy
CMD="$CMD \
    --prune_strategy \"$PRUNE_STRATEGY\""

# Add total_prune_percent
CMD="$CMD \
    --total_prune_percent $TOTAL_PRUNE_PERCENT"

# Add verbose flag if enabled
if [ "$VERBOSE" = true ]; then
    CMD="$CMD \
    --verbose"
fi

# Always add generation (assuming it's required)
CMD="$CMD \
    --generation $GENERATION"

# Add perplexity evaluation flags if enabled
if [ "$EVAL_PERPLEXITY" = true ]; then
    CMD="$CMD \
    --eval_perplexity \
    --ppl_datasets $PPL_DATASETS \
    --ppl_subjects $PPL_SUBJECTS"
    
    if [ -n "$PPL_MAX_SAMPLES" ]; then
        CMD="$CMD \
    --ppl_max_samples $PPL_MAX_SAMPLES"
    fi
fi

# Add MMLU evaluation flags if enabled
if [ "$EVAL_MMLU" = true ]; then
    CMD="$CMD \
    --eval_mmlu \
    --mmlu_datasets $MMLU_DATASETS \
    --mmlu_shots $MMLU_SHOTS"
    
    if [ -n "$MMLU_MAX_SAMPLES" ]; then
        CMD="$CMD \
    --mmlu_max_samples $MMLU_MAX_SAMPLES"
    fi
fi

# Add general NLP evaluation flags if enabled
if [ "$EVAL_GENERAL_NLP" = true ]; then
    CMD="$CMD \
    --eval_general_nlp"
    
    if [ -n "$GENERAL_NLP_DATASETS" ]; then
        CMD="$CMD \
    --general_nlp_datasets $GENERAL_NLP_DATASETS"
    fi
    
    if [ -n "$GENERAL_NLP_MAX_SAMPLES" ]; then
        CMD="$CMD \
    --general_nlp_max_samples $GENERAL_NLP_MAX_SAMPLES"
    fi
fi

CMD="$CMD \
    --save_res_dir \"$RESULTS_DIR\""

# Add save_model if set
if [ -n "$SAVE_MODEL" ]; then
    CMD="$CMD \
    --save_model \"$SAVE_MODEL\""
fi

echo "" | tee -a "$TIMING_LOG"
echo "Command:" | tee -a "$TIMING_LOG"
echo "$CMD" | tee -a "$TIMING_LOG"
echo "" | tee -a "$TIMING_LOG"

# ============================================================================
# RUN EXPERIMENT
# ============================================================================

echo "Running experiment..." | tee -a "$TIMING_LOG"
echo "Output will be saved to: $OUTPUT_LOG" | tee -a "$TIMING_LOG"
echo "" | tee -a "$TIMING_LOG"

# Run with time tracking
{ time eval "$CMD"; } 2>&1 | tee "$OUTPUT_LOG"

# Capture exit status
EXIT_STATUS=${PIPESTATUS[0]}

# ============================================================================
# FINISH AND SAVE TIMING - ALL IN ONE FILE
# ============================================================================

END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
END_EPOCH=$(date +%s)
DURATION=$((END_EPOCH - START_EPOCH))
DURATION_MIN=$((DURATION / 60))
DURATION_SEC=$((DURATION % 60))

# Extract Python timing information from output
TIMING_START=$(grep "TIMING_START=" "$OUTPUT_LOG" | tail -1 | cut -d'=' -f2)
TIMING_MODEL_LOAD=$(grep "TIMING_MODEL_LOAD=" "$OUTPUT_LOG" | tail -1 | cut -d'=' -f2)
TIMING_PREFILL=$(grep "TIMING_PREFILL=" "$OUTPUT_LOG" | tail -1 | cut -d'=' -f2)
TIMING_GENERATION=$(grep "TIMING_GENERATION=" "$OUTPUT_LOG" | tail -1 | cut -d'=' -f2)
TIMING_EVAL_PPL=$(grep "TIMING_EVAL_PPL=" "$OUTPUT_LOG" | tail -1 | cut -d'=' -f2)
TIMING_EVAL_MMLU=$(grep "TIMING_EVAL_MMLU=" "$OUTPUT_LOG" | tail -1 | cut -d'=' -f2)
TIMING_EVAL_GENERAL_NLP=$(grep "TIMING_EVAL_GENERAL_NLP=" "$OUTPUT_LOG" | tail -1 | cut -d'=' -f2)
TIMING_END=$(grep "TIMING_END=" "$OUTPUT_LOG" | tail -1 | cut -d'=' -f2)
TIMING_TOTAL=$(grep "TIMING_TOTAL=" "$OUTPUT_LOG" | tail -1 | cut -d'=' -f2)

# Write everything to the SINGLE timing log file
{
    echo ""
    echo "========================================================================"
    echo "EXPERIMENT COMPLETED"
    echo "========================================================================"
    echo "Finished at: $END_TIME"
    echo "Duration: ${DURATION_MIN}m ${DURATION_SEC}s ($DURATION seconds total)"
    echo "Exit Status: $EXIT_STATUS"
    echo ""
    
    # Bash timing from `time` command
    echo "========================================================================"
    echo "BASH TIMING (from 'time' command)"
    echo "========================================================================"
    grep -E "^(real|user|sys)" "$OUTPUT_LOG" | tail -3
    echo ""
    
    # Python timing breakdown
    if [ -n "$TIMING_START" ]; then
        echo "========================================================================"
        echo "PYTHON TIMING BREAKDOWN"
        echo "========================================================================"
        printf "%-25s %15s\n" "Phase" "Time (seconds)"
        echo "------------------------------------------------------------------------"
        printf "%-25s %15s\n" "Start timestamp" "${TIMING_START}"
        printf "%-25s %15s\n" "Model load" "${TIMING_MODEL_LOAD}"
        printf "%-25s %15s\n" "Prefill" "${TIMING_PREFILL}"
        printf "%-25s %15s\n" "Generation" "${TIMING_GENERATION}"
        printf "%-25s %15s\n" "Eval: Perplexity" "${TIMING_EVAL_PPL}"
        printf "%-25s %15s\n" "Eval: MMLU" "${TIMING_EVAL_MMLU}"
        printf "%-25s %15s\n" "Eval: General NLP" "${TIMING_EVAL_GENERAL_NLP}"
        printf "%-25s %15s\n" "End timestamp" "${TIMING_END}"
        echo "------------------------------------------------------------------------"
        printf "%-25s %15s\n" "TOTAL" "${TIMING_TOTAL}"
        echo ""
        
        # Calculate percentages
        if [ -n "$TIMING_TOTAL" ] && [ "$(echo "$TIMING_TOTAL > 0" | bc -l 2>/dev/null)" = "1" ]; then
            echo "Time Distribution:"
            echo "------------------------------------------------------------------------"
            # Use -c with proper escaping instead of heredoc
            python3 -c "
import sys
try:
    model_load = float('${TIMING_MODEL_LOAD}' if '${TIMING_MODEL_LOAD}' else '0')
    prefill = float('${TIMING_PREFILL}' if '${TIMING_PREFILL}' else '0')
    generation = float('${TIMING_GENERATION}' if '${TIMING_GENERATION}' else '0')
    eval_ppl = float('${TIMING_EVAL_PPL}' if '${TIMING_EVAL_PPL}' else '0')
    eval_mmlu = float('${TIMING_EVAL_MMLU}' if '${TIMING_EVAL_MMLU}' else '0')
    eval_general_nlp = float('${TIMING_EVAL_GENERAL_NLP}' if '${TIMING_EVAL_GENERAL_NLP}' else '0')
    total = float('${TIMING_TOTAL}' if '${TIMING_TOTAL}' else '0')
    
    if total > 0:
        print(f'  Model load:        {model_load:10.2f}s  ({model_load/total*100:5.1f}%)')
        print(f'  Prefill:           {prefill:10.2f}s  ({prefill/total*100:5.1f}%)')
        print(f'  Generation:        {generation:10.2f}s  ({generation/total*100:5.1f}%)')
        print(f'  Eval (Perplexity): {eval_ppl:10.2f}s  ({eval_ppl/total*100:5.1f}%)')
        print(f'  Eval (MMLU):       {eval_mmlu:10.2f}s  ({eval_mmlu/total*100:5.1f}%)')
        print(f'  Eval (General):    {eval_general_nlp:10.2f}s  ({eval_general_nlp/total*100:5.1f}%)')
except Exception as e:
    print(f'Error calculating percentages: {e}', file=sys.stderr)
" 2>/dev/null
            echo ""
        fi
    fi
    

    echo ""
    echo "========================================================================"
    echo "FILES SAVED"
    echo "========================================================================"
    echo "  Output log:  $OUTPUT_LOG"
    echo "  Timing log:  $TIMING_LOG"
    echo "  Config:      $CONFIG_LOG"
    echo "========================================================================"
    
} | tee -a "$TIMING_LOG"

# Create symlinks to latest files
ln -sf "output_${TIMESTAMP}.log" "$LATEST_LINK"
ln -sf "timing_${TIMESTAMP}.log" "$LATEST_TIMING"

# ============================================================================
# PRINT SUMMARY
# ============================================================================

echo ""
echo "========================================" 
echo "QUICK ACCESS"
echo "========================================" 
echo "View output:"
echo "  cat $OUTPUT_LOG"
echo "  # or: cat $LATEST_LINK"
echo ""
echo "View timing:"
echo "  cat $TIMING_LOG"
echo "  # or: cat $LATEST_TIMING"
echo ""
echo "View results directory:"
echo "  ls -lh $RESULTS_DIR"
echo "========================================" 

exit $EXIT_STATUS