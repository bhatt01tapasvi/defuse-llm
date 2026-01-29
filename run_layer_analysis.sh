#!/bin/bash

#############################################################################
# Layer Analysis Wrapper Script
# Usage: ./run_layer_analysis.sh
# 
# Features:
# - Logs all output to files
# - Tracks start/end times
# - Saves experiment configuration
# - Creates organized result directories
# - Runs systematic layer-wise pruning experiments
#############################################################################

# ============================================================================
# CONFIGURATION - Modify these parameters for different experiments
# ============================================================================

DEVICE=4                # CUDA device ID

# Model configuration
MODEL="meta-llama/Llama-3.2-3B" # Options: "gpt2", "gpt2-xl", "meta-llama/Llama-3.1-8B", etc.
NUM_LAYERS=28              # Leave empty for auto-detection (12 for GPT2, 32 for Llama-3.1-8B)

# MODEL="meta-llama/Llama-3.1-8B"
# NUM_LAYERS=32              # Leave empty for auto-detection (12 for GPT2, 32 for Llama-3.1-8B)

#MODEL="meta-llama/Llama-2-7b-hf" # Options: "gpt2", "gpt2-xl", "meta-llama/Llama-3.1-8B", etc.
#NUM_LAYERS=32 

# MODEL="meta-llama/Llama-2-13b-hf" # Options: "gpt2", "gpt2-xl", "meta-llama/Llama-3.1-8B", etc.
# NUM_LAYERS=40 

CACHE_DIR="llm_weights"

# Prompt configuration
PROMPT_TYPE="mmlu"           # Options: "custom", "mmlu"
#college_computer_science_corpus abstract_algebra_corpus high_school_biology_corpus high_school_world_history_corpus marketing_corpus philosophy_corpus professional_law_corpus
PROMPT_SUBJECTS="astronomy business_ethics college_computer_science college_mathematics world_religions high_school_mathematics econometrics global_facts electrical_engineering high_school_statistics formal_logic abstract_algebra professional_accounting international_law high_school_biology high_school_world_history marketing philosophy professional_law professional_medicine"  # Space-separated list of subjects
#PROMPT_SUBJECTS="imc_key imc_synonym_key imc_definition imc_definition_v2 imc_definition_v3 imc_synonym imc_technical imc_para imc_layman imc_analogy imc_detail imc_detail_v2 imc_application"  # Space-separated list of subjects
                               # For custom: "imc pizzas actress", etc.
                               # For mmlu: "college_computer_science abstract_algebra", etc.
PROMPT_LENGTH=500             # Leave empty for None (no prompt length limit)
CUSTOM_PROMPT_TEXT=""          # Custom text (overrides PROMPT_SUBJECT if set)

# Layer specification
LAYERS="all"                   # Options: 
                               #   "all" - test all layers
                               #   "0,1,2,5-8" - individual layers (will test each separately)
                               #   "(0-4),(5-9),(10-14),(15-19),(20-24),(25-27)"   - groups (will test each group)

# Pruning configuration
KEEP_RATES="0.5"               # Space-separated keep rates (proportion to KEEP)
                               # E.g., "0.9 0.7 0.5 0.3 0.1"
BASE_KEEP_RATE=""              # For marginal analysis: base rate for all layers
                               # If set (e.g., "0.5"), will test varying one layer at a time
                               # while keeping others at this rate
MASKING_STEP=0                 # Step at which to start masking
GENERATION=0                   # Number of tokens to generate
EMA_DECAY=""
RANKING_METHOD="max"    # Method to rank neurons for pruning - max, mean, combined, product
PRUNE_STRATEGY="topk"        # Pruning strategy - topk, auto

# Experiment control
SKIP_BASELINE=true             # Set to true to skip baseline run
EXPERIMENT_NAME=""             # Leave empty for auto-generated name
SAVE_ACTIVATIONS=false         # Set to true to save activations during model run

# Evaluation configuration - Perplexity
EVAL_PERPLEXITY=false           # Set to true to enable perplexity evaluation
#PPL_DATASETS="custom mmlu"     # Options: "custom" "mmlu" (space-separated)
#PPL_SUBJECTS="abstract_algebra_corpus anne_corpus college_computer_science_corpus food_corpus high_school_biology_corpus high_school_world_history_corpus marketing_corpus philosophy_corpus professional_law_corpus college_computer_science abstract_algebra high_school_biology high_school_world_history marketing philosophy professional_law"
PPL_DATASETS="wikitext2"     # Options: "custom" "mmlu" (space-separated)
#PPL_SUBJECTS="abstract_algebra_corpus anne_corpus college_computer_science_corpus food_corpus high_school_biology_corpus high_school_world_history_corpus marketing_corpus philosophy_corpus professional_law_corpus college_computer_science abstract_algebra high_school_biology high_school_world_history"
PPL_SUBJECTS="internet"
PPL_MAX_SAMPLES=2048            # Max samples for perplexity eval (leave empty for all)

# Evaluation configuration - MMLU
EVAL_MMLU=true                 # Set to true to enable MMLU evaluation
MMLU_DATASETS="astronomy business_ethics college_computer_science college_mathematics world_religions high_school_mathematics econometrics global_facts electrical_engineering high_school_statistics formal_logic abstract_algebra professional_accounting international_law high_school_biology high_school_world_history marketing philosophy professional_law professional_medicine"
MMLU_SHOTS=5                   # Number of few-shot examples
MMLU_MAX_SAMPLES="100"           # Max samples per MMLU task (leave empty for all)

# Evaluation configuration - General NLP
EVAL_GENERAL_NLP=false         # Set to true to enable general NLP evaluation
GENERAL_NLP_DATASETS=""        # Options: "boolq rte hellaswag winogrande arc_easy arc_challenge openbookqa"
GENERAL_NLP_MAX_SAMPLES=""     # Max samples for general NLP eval (leave empty for all)

# Results directory
BASE_RESULTS_DIR="/users/grad/abhishektyagi/wanda/wanda/results/layer_analysis/final_experiments/heat_maps"

# ============================================================================
# SETUP
# ============================================================================

# Create base results directory
mkdir -p "$BASE_RESULTS_DIR"

# Generate timestamp
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
SIGNATURE="rank_${RANKING_METHOD}_${PRUNE_STRATEGY}_ema_${EMA_DECAY}"
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
START_EPOCH=$(date +%s)

# Determine experiment name
if [ -z "$EXPERIMENT_NAME" ]; then
    MODEL_SAFE_NAME=$(echo "$MODEL" | sed 's/\//_/g')
    if [ -n "$BASE_KEEP_RATE" ]; then
        MODE="marginal_${BASE_KEEP_RATE}"
    else
        MODE="single"
    fi
    # Check if group mode
    if [[ "$LAYERS" == *"("* ]] && [[ "$LAYERS" == *")"* ]]; then
        MODE="${MODE}_group"
    fi
    EXPERIMENT_NAME="${MODEL_SAFE_NAME}_${MODE}_${TIMESTAMP}"
    #EXPERIMENT_NAME="${MODEL_SAFE_NAME}_${MODE}_${SIGNATURE}"
fi

RESULTS_DIR="${BASE_RESULTS_DIR}/${EXPERIMENT_NAME}"

# Define log files
OUTPUT_LOG="${RESULTS_DIR}/output_${TIMESTAMP}.log"
TIMING_LOG="${RESULTS_DIR}/timing_${TIMESTAMP}.log"
CONFIG_LOG="${RESULTS_DIR}/config_${TIMESTAMP}.json"
LATEST_LINK="${RESULTS_DIR}/latest_run.log"
LATEST_TIMING="${RESULTS_DIR}/latest_timing.log"

# Create results directory
mkdir -p "$RESULTS_DIR"

# ============================================================================
# PRINT CONFIGURATION
# ============================================================================

print_config() {
    echo "========================================" 
    echo "LAYER ANALYSIS CONFIGURATION"
    echo "========================================" 
    echo "Experiment Name: $EXPERIMENT_NAME"
    echo "Model: $MODEL"
    echo "Number of Layers: ${NUM_LAYERS:-Auto-detect}"
    echo ""
    echo "Layer Specification:"
    echo "  Layers: $LAYERS"
    echo "  Keep Rates: $KEEP_RATES"
    echo "  Base Keep Rate: ${BASE_KEEP_RATE:-None (single layer mode)}"
    echo "  Masking Step: $MASKING_STEP"
    echo "  EMA Decay: ${EMA_DECAY:-None (L2 norm)}"
    echo "  Ranking Method: $RANKING_METHOD"
    echo "  Prune Strategy: $PRUNE_STRATEGY"
    echo "  Skip Baseline: $SKIP_BASELINE"
    echo ""
    echo "Prompt Configuration:"
    echo "  Type: $PROMPT_TYPE"
    echo "  Subjects: $PROMPT_SUBJECTS"
    echo "  Length: ${PROMPT_LENGTH:-None}"
    if [ -n "$CUSTOM_PROMPT_TEXT" ]; then
        echo "  Custom Text: ${CUSTOM_PROMPT_TEXT:0:50}..."
    fi
    echo "  Generation Tokens: $GENERATION"
    echo "  Save Activations: $SAVE_ACTIVATIONS"
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

# Print to terminal and timing log
print_config | tee "$TIMING_LOG"

# ============================================================================
# SAVE CONFIGURATION TO JSON
# ============================================================================

# Helper function to convert empty string to null for JSON
to_json_value() {
    if [ -z "$1" ]; then
        echo "null"
    elif [[ "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        echo "$1"
    elif [ "$1" = "true" ] || [ "$1" = "false" ]; then
        echo "$1"
    else
        echo "\"$1\""
    fi
}

# Convert values for JSON
PROMPT_LENGTH_JSON=$(to_json_value "$PROMPT_LENGTH")
CUSTOM_PROMPT_TEXT_JSON=$(to_json_value "$CUSTOM_PROMPT_TEXT")
NUM_LAYERS_JSON=$(to_json_value "$NUM_LAYERS")
BASE_KEEP_RATE_JSON=$(to_json_value "$BASE_KEEP_RATE")
PPL_MAX_SAMPLES_JSON=$(to_json_value "$PPL_MAX_SAMPLES")
MMLU_MAX_SAMPLES_JSON=$(to_json_value "$MMLU_MAX_SAMPLES")
GENERAL_NLP_MAX_SAMPLES_JSON=$(to_json_value "$GENERAL_NLP_MAX_SAMPLES")
EXPERIMENT_NAME_JSON=$(to_json_value "$EXPERIMENT_NAME")
EMA_DECAY_JSON=$(to_json_value "$EMA_DECAY")

cat > "$CONFIG_LOG" << EOF
{
  "experiment_name": $EXPERIMENT_NAME_JSON,
  "timestamp": "$TIMESTAMP",
  "start_time": "$START_TIME",
  "model": "$MODEL",
  "cache_dir": "$CACHE_DIR",
  "num_layers": $NUM_LAYERS_JSON,
  "prompt": {
    "type": "$PROMPT_TYPE",
    "subjects": "$PROMPT_SUBJECTS",
    "length": $PROMPT_LENGTH_JSON,
    "custom_text": $CUSTOM_PROMPT_TEXT_JSON
  },
  "layer_analysis": {
    "layers": "$LAYERS",
    "keep_rates": "$KEEP_RATES",
    "base_keep_rate": $BASE_KEEP_RATE_JSON,
    "masking_step": $MASKING_STEP,
    "ema_decay": $EMA_DECAY_JSON,
    "ranking_method": "$RANKING_METHOD",
    "prune_strategy": "$PRUNE_STRATEGY",
    "generation": $GENERATION,
    "skip_baseline": $SKIP_BASELINE,
    "save_activations": $SAVE_ACTIVATIONS
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

CMD="CUDA_VISIBLE_DEVICES=$DEVICE stdbuf -oL -eL python -u layer_analysis.py \
    --model \"$MODEL\" \
    --cache_dir \"$CACHE_DIR\" \
    --layers \"$LAYERS\" \
    --keep_rates $KEEP_RATES \
    --masking_step $MASKING_STEP \
    --ranking_method \"$RANKING_METHOD\" \
    --prune_strategy \"$PRUNE_STRATEGY\" \
    --generation $GENERATION \
    --prompt_type \"$PROMPT_TYPE\" \
    --prompt_subject $PROMPT_SUBJECTS \
    --verbose"

# Add optional arguments
if [ -n "$NUM_LAYERS" ]; then
    CMD="$CMD \
    --num_layers $NUM_LAYERS"
fi

if [ -n "$EMA_DECAY" ]; then
    CMD="$CMD \
    --ema_decay $EMA_DECAY"
fi

if [ -n "$PROMPT_LENGTH" ]; then
    CMD="$CMD \
    --prompt_length $PROMPT_LENGTH"
fi

if [ -n "$CUSTOM_PROMPT_TEXT" ]; then
    CMD="$CMD \
    --custom_prompt_text \"$CUSTOM_PROMPT_TEXT\""
fi

if [ -n "$BASE_KEEP_RATE" ]; then
    CMD="$CMD \
    --base_keep_rate $BASE_KEEP_RATE"
fi

if [ "$SKIP_BASELINE" = true ]; then
    CMD="$CMD \
    --skip_baseline"
fi

if [ "$SAVE_ACTIVATIONS" = true ]; then
    CMD="$CMD \
    --save_activations"
fi

if [ -n "$RESULTS_DIR" ]; then
    CMD="$CMD \
    --parent_exp_dir \"$RESULTS_DIR\""
fi

# Add perplexity evaluation flags
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

# Add MMLU evaluation flags
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

# Add general NLP evaluation flags
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

echo "" | tee -a "$TIMING_LOG"
echo "Command:" | tee -a "$TIMING_LOG"
echo "$CMD" | tee -a "$TIMING_LOG"
echo "" | tee -a "$TIMING_LOG"

# ============================================================================
# RUN EXPERIMENT
# ============================================================================

echo "Running layer analysis..." | tee -a "$TIMING_LOG"
echo "Output will be saved to: $OUTPUT_LOG" | tee -a "$TIMING_LOG"
echo "" | tee -a "$TIMING_LOG"

# Run with time tracking
{ time eval "$CMD"; } 2>&1 | tee "$OUTPUT_LOG"

# Capture exit status
EXIT_STATUS=${PIPESTATUS[0]}

# ============================================================================
# FINISH AND SAVE TIMING
# ============================================================================

END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
END_EPOCH=$(date +%s)
DURATION=$((END_EPOCH - START_EPOCH))
DURATION_MIN=$((DURATION / 60))
DURATION_SEC=$((DURATION % 60))
DURATION_HOURS=$((DURATION / 3600))
DURATION_REMAINING_MIN=$(((DURATION % 3600) / 60))

# Count completed experiments from output
COMPLETED_EXPERIMENTS=$(grep -c "Progress: .* experiments completed" "$OUTPUT_LOG" || echo "0")
TOTAL_EXPERIMENTS=$(grep "Total experiments:" "$OUTPUT_LOG" | tail -1 | awk '{print $3}' || echo "Unknown")

# Write summary to timing log
{
    echo ""
    echo "========================================================================"
    echo "LAYER ANALYSIS COMPLETED"
    echo "========================================================================"
    echo "Finished at: $END_TIME"
    if [ $DURATION -ge 3600 ]; then
        echo "Duration: ${DURATION_HOURS}h ${DURATION_REMAINING_MIN}m ${DURATION_SEC}s" "($DURATION seconds total)"
    else
        echo "Duration: ${DURATION_MIN}m ${DURATION_SEC}s" "($DURATION seconds total)"
    fi
    echo "Exit Status: $EXIT_STATUS"
    echo ""
    echo "Experiments Completed: ${COMPLETED_EXPERIMENTS}/${TOTAL_EXPERIMENTS}"
    echo ""
    
    # Bash timing from `time` command
    echo "========================================================================"
    echo "BASH TIMING (from 'time' command)"
    echo "========================================================================"
    grep -E "^(real|user|sys)" "$OUTPUT_LOG" | tail -3
    echo ""
    
    # Extract per-experiment timing if available
    echo "========================================================================"
    echo "EXPERIMENT PROGRESS"
    echo "========================================================================"
    grep "Progress: .* experiments completed" "$OUTPUT_LOG" | tail -10
    echo ""
    
    echo "========================================================================"
    echo "FILES SAVED"
    echo "========================================================================"
    echo "  Output log:  $OUTPUT_LOG"
    echo "  Timing log:  $TIMING_LOG"
    echo "  Config:      $CONFIG_LOG"
    echo "  Results dir: $RESULTS_DIR"
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
echo ""
echo "View experiment results:"
echo "  find $RESULTS_DIR -name 'perplexity_results.json'"
echo "  find $RESULTS_DIR -name 'results_mmlu_*.json'"
echo "========================================" 

exit $EXIT_STATUS