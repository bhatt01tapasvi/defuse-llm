#!/bin/bash

#############################################################################
# Multi-Subject Layer Analysis Script
# Usage: ./run_subject_layer_analysis.sh subject1 subject2 subject3 ...
# 
# Tests each layer individually for each subject by:
# 1. Using a specific MMLU subject as the prompt
# 2. Evaluating MMLU accuracy on that same subject after pruning each layer
# 3. Running this for all layers to see which layers are most important
#
# Example:
#   ./run_subject_layer_analysis.sh astronomy business_ethics college_computer_science
#############################################################################

# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE=7                # CUDA device ID

# Model configuration
MODEL="meta-llama/Llama-3.2-3B"
NUM_LAYERS=28

# Cache directory
CACHE_DIR="llm_weights"

# Subjects to test (from command line arguments)
SUBJECTS=("$@")

# Layer specification - test each layer individually
LAYERS="all"                # Will test layers 0 through NUM_LAYERS-1 individually

# Pruning configuration
KEEP_RATES="0.3"    # Test multiple pruning levels
MASKING_STEP=0
GENERATION=0
RANKING_METHOD="magnitude"
PRUNE_STRATEGY="topk"

# Experiment control
SKIP_BASELINE=false         # Run baseline to compare against
SAVE_ACTIVATIONS=false

# MMLU Evaluation settings
MMLU_SHOTS=5
MMLU_MAX_SAMPLES=100        # Number of samples to evaluate per layer

# Results directory
BASE_RESULTS_DIR="/users/grad/abhishektyagi/wanda/wanda/results/layer_analysis/subject_specific_v3"

# ============================================================================
# SETUP
# ============================================================================

# If no subjects provided, use default list
if [ ${#SUBJECTS[@]} -eq 0 ]; then
    SUBJECTS=(astronomy business_ethics college_computer_science college_mathematics world_religions)
fi

TOTAL_SUBJECTS=${#SUBJECTS[@]}

# Create base results directory
mkdir -p "$BASE_RESULTS_DIR"

# Generate main timestamp
MAIN_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

echo "========================================" 
echo "MULTI-SUBJECT LAYER ANALYSIS"
echo "========================================" 
echo "Total subjects to process: $TOTAL_SUBJECTS"
echo "Subjects: ${SUBJECTS[*]}"
echo "Model: $MODEL"
echo "Layers: 0-$((NUM_LAYERS-1))"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================" 
echo ""

# Track success/failure
SUCCESSFUL=0
FAILED=0
FAILED_SUBJECTS=()

# ============================================================================
# FUNCTIONS FOR RUNNING EACH SUBJECT
# ============================================================================

run_subject_analysis() {
    local SUBJECT="$1"
    local SUBJECT_NUM="$2"
    
    echo ""
    echo "========================================================================"
    echo "PROCESSING SUBJECT $SUBJECT_NUM/$TOTAL_SUBJECTS: $SUBJECT"
    echo "========================================================================"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Generate timestamp and experiment name for this subject
    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    MODEL_SAFE_NAME=$(echo "$MODEL" | sed 's/\//_/g')
    EXPERIMENT_NAME="${MODEL_SAFE_NAME}_subject_${SUBJECT}_${TIMESTAMP}"
    RESULTS_DIR="${BASE_RESULTS_DIR}/${EXPERIMENT_NAME}"
    
    # Define log files
    OUTPUT_LOG="${RESULTS_DIR}/output_${TIMESTAMP}.log"
    TIMING_LOG="${RESULTS_DIR}/timing_${TIMESTAMP}.log"
    CONFIG_LOG="${RESULTS_DIR}/config_${TIMESTAMP}.json"
    
    # Create results directory
    mkdir -p "$RESULTS_DIR"
    
    # Print configuration
    {
        echo "========================================" 
        echo "SUBJECT-SPECIFIC LAYER ANALYSIS"
        echo "========================================" 
        echo "Experiment Name: $EXPERIMENT_NAME"
        echo "Model: $MODEL"
        echo "Number of Layers: $NUM_LAYERS"
        echo "Subject: $SUBJECT"
        echo ""
        echo "Configuration:"
        echo "  Testing: Each layer individually (0 to $((NUM_LAYERS-1)))"
        echo "  Prompt: MMLU ${SUBJECT}"
        echo "  Evaluation: MMLU ${SUBJECT}"
        echo "  Keep Rates: $KEEP_RATES"
        echo "  MMLU Shots: $MMLU_SHOTS"
        echo "  Max Samples: $MMLU_MAX_SAMPLES"
        echo "  Ranking Method: $RANKING_METHOD"
        echo ""
        echo "Results Directory: $RESULTS_DIR"
        echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================" 
    } | tee "$TIMING_LOG"
    
    # Save configuration to JSON
    cat > "$CONFIG_LOG" << EOF
{
  "experiment_name": "$EXPERIMENT_NAME",
  "timestamp": "$TIMESTAMP",
  "start_time": "$(date '+%Y-%m-%d %H:%M:%S')",
  "model": "$MODEL",
  "num_layers": $NUM_LAYERS,
  "subject": "$SUBJECT",
  "layer_analysis": {
    "layers": "$LAYERS",
    "keep_rates": "$KEEP_RATES",
    "masking_step": $MASKING_STEP,
    "ranking_method": "$RANKING_METHOD",
    "prune_strategy": "$PRUNE_STRATEGY",
    "skip_baseline": $SKIP_BASELINE
  },
  "mmlu_evaluation": {
    "shots": $MMLU_SHOTS,
    "max_samples": $MMLU_MAX_SAMPLES
  },
  "results_dir": "$RESULTS_DIR"
}
EOF
    
    echo "Configuration saved to: $CONFIG_LOG" | tee -a "$TIMING_LOG"
    
    # Build command
    CMD="CUDA_VISIBLE_DEVICES=$DEVICE python -u layer_analysis.py \
        --model \"$MODEL\" \
        --cache_dir \"$CACHE_DIR\" \
        --num_layers $NUM_LAYERS \
        --layers \"$LAYERS\" \
        --keep_rates $KEEP_RATES \
        --masking_step $MASKING_STEP \
        --ranking_method \"$RANKING_METHOD\" \
        --prune_strategy \"$PRUNE_STRATEGY\" \
        --generation $GENERATION \
        --prompt_type mmlu \
        --prompt_subject $SUBJECT \
        --prompt_length 500 \
        --eval_mmlu \
        --mmlu_datasets $SUBJECT \
        --mmlu_shots $MMLU_SHOTS \
        --mmlu_max_samples $MMLU_MAX_SAMPLES \
        --parent_exp_dir \"$RESULTS_DIR\" \
        --verbose"
    
    if [ "$SKIP_BASELINE" = true ]; then
        CMD="$CMD --skip_baseline"
    fi
    
    if [ "$SAVE_ACTIVATIONS" = true ]; then
        CMD="$CMD --save_activations"
    fi
    
    echo "" | tee -a "$TIMING_LOG"
    echo "Command:" | tee -a "$TIMING_LOG"
    echo "$CMD" | tee -a "$TIMING_LOG"
    echo "" | tee -a "$TIMING_LOG"
    
    echo "Running analysis..." | tee -a "$TIMING_LOG"
    echo "This will test each layer (0-$((NUM_LAYERS-1))) individually" | tee -a "$TIMING_LOG"
    echo "Output: $OUTPUT_LOG" | tee -a "$TIMING_LOG"
    echo "" | tee -a "$TIMING_LOG"
    
    START_EPOCH=$(date +%s)
    
    # Run with time tracking
    { time eval "$CMD"; } 2>&1 | tee "$OUTPUT_LOG"
    
    EXIT_STATUS=${PIPESTATUS[0]}
    
    # Finish and save timing
    END_EPOCH=$(date +%s)
    DURATION=$((END_EPOCH - START_EPOCH))
    DURATION_MIN=$((DURATION / 60))
    DURATION_SEC=$((DURATION % 60))
    
    {
        echo ""
        echo "========================================================================"
        echo "ANALYSIS COMPLETED FOR SUBJECT: $SUBJECT"
        echo "========================================================================"
        echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Duration: ${DURATION_MIN}m ${DURATION_SEC}s"
        echo "Exit Status: $EXIT_STATUS"
        echo ""
        echo "Results saved to: $RESULTS_DIR"
        echo "========================================================================"
    } | tee -a "$TIMING_LOG"
    
    echo ""
    echo "------------------------------------------------------------------------"
    if [ $EXIT_STATUS -eq 0 ]; then
        echo "✓ Subject '$SUBJECT' completed successfully"
    else
        echo "✗ Subject '$SUBJECT' failed (exit code: $EXIT_STATUS)"
    fi
    echo "------------------------------------------------------------------------"
    echo ""
    
    return $EXIT_STATUS
}

# ============================================================================
# MAIN LOOP - RUN ANALYSIS FOR EACH SUBJECT
# ============================================================================

for i in "${!SUBJECTS[@]}"; do
    SUBJECT="${SUBJECTS[$i]}"
    SUBJECT_NUM=$((i + 1))
    
    run_subject_analysis "$SUBJECT" "$SUBJECT_NUM"
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        ((SUCCESSFUL++))
    else
        ((FAILED++))
        FAILED_SUBJECTS+=("$SUBJECT")
    fi
    
    # Brief pause between experiments
    if [ $SUBJECT_NUM -lt $TOTAL_SUBJECTS ]; then
        echo "Waiting 5 seconds before next subject..."
        sleep 5
    fi
done

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo ""
echo "========================================================================"
echo "ALL SUBJECTS COMPLETED"
echo "========================================================================"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Summary:"
echo "  Total subjects:    $TOTAL_SUBJECTS"
echo "  Successful:        $SUCCESSFUL"
echo "  Failed:            $FAILED"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failed subjects:"
    for subject in "${FAILED_SUBJECTS[@]}"; do
        echo "  - $subject"
    done
fi

echo ""
echo "All results saved to: $BASE_RESULTS_DIR"
echo "========================================================================"
echo ""

# Exit with error if any experiments failed
if [ $FAILED -eq 0 ]; then
    exit 0
else
    exit 1
fi
