#!/bin/bash

#############################################################################
# Multi-Subject Experiment Runner
# Usage: ./run_experiment_multi_subject.sh subject1 subject2 subject3 ...
# 
# Example:
#   ./run_experiment_multi_subject.sh astronomy business_ethics college_computer_science
#
# This script runs run_experiment.sh for each subject automatically
#############################################################################

# Check if subjects were provided
if [ $# -eq 0 ]; then
    echo "Error: No subjects specified!"
    echo ""
    echo "Usage: $0 subject1 subject2 subject3 ..."
    echo ""
    echo "Example:"
    echo "  $0 astronomy business_ethics college_computer_science"
    echo ""
    echo "Available MMLU subjects:"
    echo "  astronomy, business_ethics, college_computer_science, college_mathematics,"
    echo "  world_religions, high_school_mathematics, econometrics, global_facts,"
    echo "  electrical_engineering, high_school_statistics, formal_logic, abstract_algebra,"
    echo "  professional_accounting, international_law, high_school_biology,"
    echo "  high_school_world_history, marketing, philosophy, professional_law,"
    echo "  professional_medicine, and more..."
    echo ""
    echo "Or run with no arguments to use the default subject list (20 subjects)"
    exit 1
fi

# Store all subjects (from command line arguments)
SUBJECTS=("$@")

# If no subjects provided, use default list
#Add astronomy later!
if [ ${#SUBJECTS[@]} -eq 0 ]; then
    SUBJECTS=(astronomy business_ethics college_computer_science college_mathematics world_religions high_school_mathematics econometrics global_facts electrical_engineering high_school_statistics formal_logic abstract_algebra professional_accounting international_law high_school_biology high_school_world_history marketing philosophy professional_law professional_medicine)
fi

TOTAL_SUBJECTS=${#SUBJECTS[@]}

echo "========================================" 
echo "MULTI-SUBJECT EXPERIMENT RUNNER"
echo "========================================" 
echo "Total subjects to process: $TOTAL_SUBJECTS"
echo "Subjects: ${SUBJECTS[*]}"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================" 
echo ""

# Track success/failure
SUCCESSFUL=0
FAILED=0
FAILED_SUBJECTS=()

# Run experiment for each subject
for i in "${!SUBJECTS[@]}"; do
    SUBJECT="${SUBJECTS[$i]}"
    SUBJECT_NUM=$((i + 1))
    
    echo ""
    echo "========================================================================"
    echo "PROCESSING SUBJECT $SUBJECT_NUM/$TOTAL_SUBJECTS: $SUBJECT"
    echo "========================================================================"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Export the subject as an environment variable that run_experiment.sh will use
    export SUBJECT_OVERRIDE="$SUBJECT"
    
    # Run the experiment
    ./run_experiment.sh
    EXIT_CODE=$?
    
    echo ""
    echo "------------------------------------------------------------------------"
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Subject '$SUBJECT' completed successfully"
        ((SUCCESSFUL++))
    else
        echo "✗ Subject '$SUBJECT' failed (exit code: $EXIT_CODE)"
        ((FAILED++))
        FAILED_SUBJECTS+=("$SUBJECT")
    fi
    echo "------------------------------------------------------------------------"
    echo ""
    
    # Brief pause between experiments
    if [ $SUBJECT_NUM -lt $TOTAL_SUBJECTS ]; then
        echo "Waiting 5 seconds before next subject..."
        sleep 5
    fi
done

# Print final summary
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

echo "========================================================================"
echo ""

# Exit with error if any experiments failed
if [ $FAILED -eq 0 ]; then
    exit 0
else
    exit 1
fi
