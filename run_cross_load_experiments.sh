#!/bin/bash
#
# Cross-Load Transfer Learning Experiments
#
# Tests generalization ability by training on one load condition
# and testing on a different load condition (Cross_load dataset).
#
# This script runs experiments for:
# 1. Train on load_0, test on Cross_load
# 2. Train on load_1, test on Cross_load
# 3. Train on load_2, test on Cross_load
# 4. Train on load_3, test on Cross_load
# 5. Train on combined (all loads), test on Cross_load
#

set -e  # Exit on error

# Configuration
CROSS_LOAD_TEST="Training_data/Cross_load/test.txt"
NUM_CLASSES=4
EPOCHS=100
BASE_OUTPUT_DIR="experiments/cross_load_transfer"

echo "================================================================================"
echo "CROSS-LOAD TRANSFER LEARNING EXPERIMENTS"
echo "================================================================================"
echo "Testing generalization across different load conditions"
echo ""

# Check if Cross_load test data exists
if [ ! -f "$CROSS_LOAD_TEST" ]; then
    echo "Error: Cross_load test data not found at $CROSS_LOAD_TEST"
    echo "Please run: python src/data/data_preparation.py"
    exit 1
fi

mkdir -p "$BASE_OUTPUT_DIR"

# Function to run a single experiment
run_experiment() {
    local TRAIN_LOAD=$1
    local TRAIN_DATA=$2
    local VAL_DATA=$3
    local EXP_NAME=$4
    
    echo ""
    echo "================================================================================"
    echo "EXPERIMENT: Train on ${TRAIN_LOAD}, Test on Cross_load"
    echo "================================================================================"
    echo ""
    
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${EXP_NAME}"
    
    # Step 1: Train Teacher on source load
    echo ">>> Step 1: Training Teacher (SE-ResNet152) on ${TRAIN_LOAD}..."
    TEACHER_OUTPUT="${OUTPUT_DIR}/teacher"
    
    if [ -f "${TEACHER_OUTPUT}/checkpoints/best_model.pth" ]; then
        echo "✓ Teacher already trained, skipping..."
    else
        python src/train_teacher.py \
            --train-data "$TRAIN_DATA" \
            --val-data "$VAL_DATA" \
            --num-classes $NUM_CLASSES \
            --epochs $EPOCHS \
            --batch-size 16 \
            --lr 0.00625 \
            --optimizer sgd \
            --scheduler multistep \
            --output-dir "$TEACHER_OUTPUT" \
            --num-workers 4
    fi
    
    TEACHER_CHECKPOINT="${TEACHER_OUTPUT}/checkpoints/best_model.pth"
    
    # Step 2: Train Baseline Student on source load
    echo ""
    echo ">>> Step 2: Training Baseline MobileNetV2 on ${TRAIN_LOAD}..."
    BASELINE_OUTPUT="${OUTPUT_DIR}/baseline_mobilenetv2"
    
    if [ -f "${BASELINE_OUTPUT}/checkpoints/best_model.pth" ]; then
        echo "✓ Baseline already trained, skipping..."
    else
        python src/train_student.py \
            --train-data "$TRAIN_DATA" \
            --val-data "$VAL_DATA" \
            --num-classes $NUM_CLASSES \
            --model mobilenetv2 \
            --distillation-mode none \
            --epochs $EPOCHS \
            --batch-size 32 \
            --lr 0.0125 \
            --optimizer sgd \
            --scheduler multistep \
            --output-dir "$BASELINE_OUTPUT" \
            --num-workers 4
    fi
    
    # Step 3: Train Distilled Student on source load
    echo ""
    echo ">>> Step 3: Training Distilled MobileNetV2 on ${TRAIN_LOAD}..."
    DISTILLED_OUTPUT="${OUTPUT_DIR}/distilled_mobilenetv2"
    
    if [ -f "${DISTILLED_OUTPUT}/checkpoints/best_model.pth" ]; then
        echo "✓ Distilled already trained, skipping..."
    else
        python src/train_student.py \
            --train-data "$TRAIN_DATA" \
            --val-data "$VAL_DATA" \
            --num-classes $NUM_CLASSES \
            --model mobilenetv2 \
            --distillation-mode online \
            --teacher-checkpoint "$TEACHER_CHECKPOINT" \
            --temperature 4.0 \
            --alpha 0.3 \
            --epochs $EPOCHS \
            --batch-size 32 \
            --lr 0.0125 \
            --optimizer sgd \
            --scheduler multistep \
            --output-dir "$DISTILLED_OUTPUT" \
            --num-workers 4
    fi
    
    # Step 4: Evaluate on Cross_load (transfer learning test)
    echo ""
    echo ">>> Step 4: Evaluating on Cross_load (Transfer Test)..."
    EVAL_OUTPUT="${OUTPUT_DIR}/evaluation_cross_load"
    
    python src/evaluate_distillation.py \
        --test-data "$CROSS_LOAD_TEST" \
        --num-classes $NUM_CLASSES \
        --teacher-checkpoint "$TEACHER_CHECKPOINT" \
        --baseline-checkpoints "${BASELINE_OUTPUT}/checkpoints/best_model.pth" \
        --distilled-checkpoints "${DISTILLED_OUTPUT}/checkpoints/best_model.pth" \
        --batch-size 32 \
        --num-workers 4 \
        --output-dir "$EVAL_OUTPUT"
    
    echo ""
    echo "✓ Experiment Complete: Train on ${TRAIN_LOAD}, Test on Cross_load"
    echo "  Results: ${EVAL_OUTPUT}/evaluation_results.json"
}

# Experiment 1: Train on load_0
if [ -f "Training_data/load_0/train.txt" ]; then
    run_experiment "load_0" \
        "Training_data/load_0/train.txt" \
        "Training_data/load_0/test.txt" \
        "train_load0_test_cross"
fi

# Experiment 2: Train on load_1
if [ -f "Training_data/load_1/train.txt" ]; then
    run_experiment "load_1" \
        "Training_data/load_1/train.txt" \
        "Training_data/load_1/test.txt" \
        "train_load1_test_cross"
fi

# Experiment 3: Train on load_2
if [ -f "Training_data/load_2/train.txt" ]; then
    run_experiment "load_2" \
        "Training_data/load_2/train.txt" \
        "Training_data/load_2/test.txt" \
        "train_load2_test_cross"
fi

# Experiment 4: Train on load_3
if [ -f "Training_data/load_3/train.txt" ]; then
    run_experiment "load_3" \
        "Training_data/load_3/train.txt" \
        "Training_data/load_3/test.txt" \
        "train_load3_test_cross"
fi

# Experiment 5: Train on combined (all loads)
if [ -f "Training_data/combined/train.txt" ]; then
    run_experiment "combined" \
        "Training_data/combined/train.txt" \
        "Training_data/combined/test.txt" \
        "train_combined_test_cross"
fi

# Summary Report
echo ""
echo "================================================================================"
echo "CROSS-LOAD TRANSFER LEARNING - SUMMARY REPORT"
echo "================================================================================"
echo ""
echo "Analyzing transfer learning performance across load conditions..."
echo ""

# Create summary report
SUMMARY_FILE="${BASE_OUTPUT_DIR}/transfer_learning_summary.txt"
echo "Cross-Load Transfer Learning Results" > "$SUMMARY_FILE"
echo "=====================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Training Load -> Cross_load Test Performance:" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Extract and display results
for exp_dir in ${BASE_OUTPUT_DIR}/train_*/; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename "$exp_dir")
        results_file="${exp_dir}/evaluation_cross_load/evaluation_results.json"
        
        if [ -f "$results_file" ]; then
            echo "Experiment: $exp_name" >> "$SUMMARY_FILE"
            echo "------------------------" >> "$SUMMARY_FILE"
            
            # Extract accuracies using Python
            python3 -c "
import json
import sys

try:
    with open('$results_file', 'r') as f:
        results = json.load(f)
    
    for model_name, metrics in results.items():
        acc = metrics.get('accuracy', 0)
        print(f'  {model_name}: {acc:.2f}%')
        sys.stdout.flush()
except Exception as e:
    print(f'  Error reading results: {e}')
" | tee -a "$SUMMARY_FILE"
            
            echo "" >> "$SUMMARY_FILE"
        fi
    fi
done

echo ""
echo "================================================================================"
echo "✓ CROSS-LOAD TRANSFER LEARNING EXPERIMENTS COMPLETED"
echo "================================================================================"
echo ""
echo "Results saved to: ${BASE_OUTPUT_DIR}/"
echo "Summary report: ${SUMMARY_FILE}"
echo ""
echo "Key Findings:"
echo "  - Compare performance across different source training loads"
echo "  - Evaluate generalization to Cross_load test set"
echo "  - Assess distillation benefits for transfer learning"
echo ""
cat "$SUMMARY_FILE"
echo ""

