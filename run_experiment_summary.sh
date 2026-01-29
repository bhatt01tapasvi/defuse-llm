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

DEVICE=1                  # CUDA device ID

# Model configuration
MODE="auto"             # Options: "manual", "auto"
MODEL="meta-llama/Llama-3.2-3B-Instruct" # Options: "gpt2", "gpt2-xl", "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.2-3B","Llama-2-7b-hf"  etc.
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

LAYER_TOPK="all:auto"
#LAYER_TOPK=$LAYER_TOPK_2_7b_85

MASKING_STEP=0
RELEASE_STEP=""              # Step at which to release masked neurons (leave empty for None)
GENERATION=0
EMA_DECAY=""
RANKING_METHOD="max"
PRUNE_STRATEGY="topk"
TOTAL_PRUNE_PERCENT=50.0
VERBOSE=true                 # Enable verbose output from NeuronDefuser
KNOWLEDGE_DRIFT=true        # Enable knowledge drift tracking

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
EVAL_GENERAL_NLP=false         # Set to true to enable general NLP evaluation
GENERAL_NLP_DATASETS="mmlu gpqa medmcqa"        # Options: "boolq rte hellaswag winogrande arc_easy arc_challenge openbookqa"
GENERAL_NLP_MAX_SAMPLES=""     # Max samples for general NLP eval (leave empty for all)

# Evaluation configuration - Summarization
EVAL_SUMMARIZATION=true      # Set to true to enable summarization evaluation
SUMMARIZATION_DATASETS="multi_news"  # Summarization datasets (space-separated)
SUMMARIZATION_MAX_SAMPLES=100  # Max samples for summarization eval (leave empty for all)
SUMMARIZATION_MAX_OUTPUT=1000  # Max output tokens for summaries

# Results directory
EXPERIMENT_NAME=${SUMMARIZATION_DATASETS}_${TOTAL_PRUNE_PERCENT}prune_kd${KNOWLEDGE_DRIFT}_threshold5_1sigma
BASE_RESULTS_DIR="/users/grad/abhishektyagi/wanda/wanda/results/knowledge_drift/benchmark"
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
    echo "  Summarization: $EVAL_SUMMARIZATION"
    if [ "$EVAL_SUMMARIZATION" = true ]; then
        echo "    - Datasets: ${SUMMARIZATION_DATASETS:-Default}"
        echo "    - Max Samples: ${SUMMARIZATION_MAX_SAMPLES:-All}"
        echo "    - Max Output Tokens: $SUMMARIZATION_MAX_OUTPUT"
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

if [ -z "$SUMMARIZATION_MAX_SAMPLES" ]; then
    SUMMARIZATION_MAX_SAMPLES_JSON="null"
else
    SUMMARIZATION_MAX_SAMPLES_JSON="$SUMMARIZATION_MAX_SAMPLES"
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
    "verbose": $VERBOSE,
    "knowledge_drift": $KNOWLEDGE_DRIFT
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
    },
    "summarization": {
      "enabled": $EVAL_SUMMARIZATION,
      "datasets": "$SUMMARIZATION_DATASETS",
      "max_samples": $SUMMARIZATION_MAX_SAMPLES_JSON,
      "max_output": $SUMMARIZATION_MAX_OUTPUT
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
# Add knowledge_drift flag if enabled
if [ "$KNOWLEDGE_DRIFT" = true ]; then
    CMD="$CMD \\
    --knowledge_drift"
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
# Add summarization evaluation flags if enabled
if [ "$EVAL_SUMMARIZATION" = true ]; then
    CMD="$CMD \\
    --eval_summarization"
    
    if [ -n "$SUMMARIZATION_DATASETS" ]; then
        CMD="$CMD \\
    --summarization_datasets $SUMMARIZATION_DATASETS"
    fi
    
    if [ -n "$SUMMARIZATION_MAX_SAMPLES" ]; then
        CMD="$CMD \\
    --summarization_max_samples $SUMMARIZATION_MAX_SAMPLES"
    fi
    
    if [ -n "$SUMMARIZATION_MAX_OUTPUT" ]; then
        CMD="$CMD \\
    --summarization_max_output $SUMMARIZATION_MAX_OUTPUT"
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