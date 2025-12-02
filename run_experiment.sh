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

# Model configuration
MODEL="meta-llama/Llama-3.1-8B" # Options: "gpt2", "gpt2-xl", "meta-llama/Llama-3.1-8B", etc.
CACHE_DIR="llm_weights"

# Prompt configuration
PROMPT_TYPE="mmlu"           # Options: "custom", "mmlu"
PROMPT_SUBJECT="college_computer_science"           # For custom: "imc", "pizzas", "actress", etc.
                               # For mmlu: "college_computer_science", etc.
PROMPT_LENGTH=""               # Leave empty for None (no prompt length limit)

# Pruning configuration
## Uncomment and set these for specific pruning, comment the set below
##LAYER_TOPK="10:0.5,11:0.5,12:0.5,13:0.5,14:0.5,15:0.5,16:0.5,17:0.5,18:0.5,19:0.5,20:0.5"
##MASKING_STEP=100
##GENERATION=150

## Comment these out when pruning.
LAYER_TOPK=""
MASKING_STEP=""
GENERATION=150

# Evaluation configuration - Perplexity
EVAL_PERPLEXITY=true           # Set to true to enable perplexity evaluation
PPL_DATASETS="mmlu"          # Options: "custom", "mmlu" (space-separated)
PPL_SUBJECTS="college_computer_science"             # Subjects for perplexity evaluation (space-separated)
                               # For custom: "imc", "food_corpus", "anne_corpus"
                               # For mmlu: "college_computer_science", "machine_learning", etc.
PPL_MAX_SAMPLES=""             # Max samples for perplexity eval (leave empty for all)

# Evaluation configuration - MMLU
EVAL_MMLU=false                # Set to true to enable MMLU evaluation
MMLU_DATASETS="college_computer_science machine_learning electrical_engineering business_ethics world_religions prehistory moral_disputes"
                               # MMLU subjects to evaluate (space-separated)
MMLU_SHOTS=0                   # Number of few-shot examples (0=zero-shot, 5=five-shot)
MMLU_MAX_SAMPLES=""            # Max samples per MMLU task (leave empty for all)

# Evaluation configuration - General NLP
EVAL_GENERAL_NLP=false         # Set to true to enable general NLP evaluation
GENERAL_NLP_DATASETS=""        # Options: "boolq rte hellaswag winogrande arc_easy arc_challenge openbookqa"
GENERAL_NLP_MAX_SAMPLES=""     # Max samples for general NLP eval (leave empty for all)

# Results directory
EXPERIMENT_NAME="prune_10to20_prompt_ccs_mmlu"
BASE_RESULTS_DIR="/users/grad/abhishektyagi/wanda/wanda/results/sanity"
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
    echo "Model: $MODEL"
    echo "Prompt Type: $PROMPT_TYPE"
    echo "Prompt Subject: $PROMPT_SUBJECT"
    echo "Prompt Length: ${PROMPT_LENGTH:-None}"
    echo "Layer TopK: $LAYER_TOPK"
    echo "Masking Step: $MASKING_STEP"
    echo "Generation Tokens: $GENERATION"
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

cat > "$CONFIG_LOG" << EOF
{
  "experiment_name": "$EXPERIMENT_NAME",
  "timestamp": "$TIMESTAMP",
  "start_time": "$START_TIME",
  "model": "$MODEL",
  "cache_dir": "$CACHE_DIR",
  "prompt": {
    "type": "$PROMPT_TYPE",
    "subject": "$PROMPT_SUBJECT",
    "length": $PROMPT_LENGTH_JSON
  },
  "pruning": {
    "layer_topk": $LAYER_TOPK_JSON,
    "masking_step": $MASKING_STEP_JSON,
    "generation": $GENERATION
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

CMD="python dynamicPrune.py \
    --model \"$MODEL\" \
    --cache_dir \"$CACHE_DIR\" \
    --prompt_type \"$PROMPT_TYPE\" \
    --prompt_subject \"$PROMPT_SUBJECT\""

# Add prompt_length only if it's set (not empty)
if [ -n "$PROMPT_LENGTH" ]; then
    CMD="$CMD \
    --prompt_length $PROMPT_LENGTH"
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
    echo "Duration: ${DURATION_MIN}m ${DURATION_SEC}s (${DURATION} seconds total)"
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
            python3 << EOF
try:
    model_load = float("$TIMING_MODEL_LOAD")
    prefill = float("$TIMING_PREFILL")
    generation = float("$TIMING_GENERATION")
    eval_ppl = float("$TIMING_EVAL_PPL")
    eval_mmlu = float("$TIMING_EVAL_MMLU")
    eval_general_nlp = float("$TIMING_EVAL_GENERAL_NLP")
    total = float("$TIMING_TOTAL")
    
    if total > 0:
        print(f"  Model load:        {model_load:10.2f}s  ({model_load/total*100:5.1f}%)")
        print(f"  Prefill:           {prefill:10.2f}s  ({prefill/total*100:5.1f}%)")
        print(f"  Generation:        {generation:10.2f}s  ({generation/total*100:5.1f}%)")
        print(f"  Eval (Perplexity): {eval_ppl:10.2f}s  ({eval_ppl/total*100:5.1f}%)")
        print(f"  Eval (MMLU):       {eval_mmlu:10.2f}s  ({eval_mmlu/total*100:5.1f}%)")
        print(f"  Eval (General):    {eval_general_nlp:10.2f}s  ({eval_general_nlp/total*100:5.1f}%)")
except:
    pass
EOF
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