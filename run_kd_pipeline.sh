#!/bin/bash
#
# Automated Knowledge Distillation Pipeline
# 
# This script runs the complete KD pipeline:
# 1. Train teacher model (SE-ResNet152)
# 2. Train baseline student models (MobileNetV2, ResNet18)
# 3. Train distilled student models with KD
# 4. Evaluate and compare all models
#
# Usage: ./run_kd_pipeline.sh [load_condition]
# Example: ./run_kd_pipeline.sh load_0
#

set -e  # Exit on error

# Configuration
LOAD_CONDITION="${1:-load_0}"
TRAIN_DATA="Training_data/${LOAD_CONDITION}/train.txt"
TEST_DATA="Training_data/${LOAD_CONDITION}/test.txt"
NUM_CLASSES=4
BASE_OUTPUT_DIR="experiments/${LOAD_CONDITION}"

# Hyperparameters
TEACHER_EPOCHS=100
STUDENT_EPOCHS=100
TEACHER_BS=16
STUDENT_BS=32
TEMPERATURE=4.0
ALPHA=0.3

echo "================================================================================"
echo "KNOWLEDGE DISTILLATION PIPELINE FOR ${LOAD_CONDITION}"
echo "================================================================================"
echo "Train data: ${TRAIN_DATA}"
echo "Test data: ${TEST_DATA}"
echo ""

# Check if data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: Training data not found at $TRAIN_DATA"
    echo "Please run: python src/data/data_preparation.py"
    exit 1
fi

if [ ! -f "$TEST_DATA" ]; then
    echo "Error: Test data not found at $TEST_DATA"
    exit 1
fi

# ============================================================================
# STEP 1: Train Teacher Model
# ============================================================================
echo ""
echo "================================================================================"
echo "STEP 1: TRAINING TEACHER MODEL (SE-ResNet152)"
echo "================================================================================"
echo ""

TEACHER_OUTPUT="${BASE_OUTPUT_DIR}/teacher"

if [ -f "${TEACHER_OUTPUT}/checkpoints/best_model.pth" ]; then
    echo "✓ Teacher model already trained. Skipping..."
    echo "  (Delete ${TEACHER_OUTPUT} to retrain)"
else
    python src/train_teacher.py \
        --train-data "$TRAIN_DATA" \
        --val-data "$TEST_DATA" \
        --num-classes $NUM_CLASSES \
        --epochs $TEACHER_EPOCHS \
        --batch-size $TEACHER_BS \
        --lr 0.00625 \
        --optimizer sgd \
        --momentum 0.9 \
        --weight-decay 1e-4 \
        --scheduler multistep \
        --gamma 0.1 \
        --output-dir "$TEACHER_OUTPUT" \
        --save-freq 10 \
        --num-workers 4
    
    echo ""
    echo "✓ Teacher training completed!"
fi

TEACHER_CHECKPOINT="${TEACHER_OUTPUT}/checkpoints/best_model.pth"

# ============================================================================
# STEP 2: Train Baseline Student Models
# ============================================================================
echo ""
echo "================================================================================"
echo "STEP 2: TRAINING BASELINE STUDENT MODELS"
echo "================================================================================"
echo ""

# MobileNetV2 Baseline
echo ">>> Training MobileNetV2 (Baseline)..."
BASELINE_MOBILENET_OUTPUT="${BASE_OUTPUT_DIR}/baseline/mobilenetv2"

if [ -f "${BASELINE_MOBILENET_OUTPUT}/checkpoints/best_model.pth" ]; then
    echo "✓ MobileNetV2 baseline already trained. Skipping..."
else
    python src/train_student.py \
        --train-data "$TRAIN_DATA" \
        --val-data "$TEST_DATA" \
        --num-classes $NUM_CLASSES \
        --model mobilenetv2 \
        --distillation-mode none \
        --epochs $STUDENT_EPOCHS \
        --batch-size $STUDENT_BS \
        --lr 0.0125 \
        --optimizer sgd \
        --momentum 0.9 \
        --weight-decay 1e-4 \
        --scheduler multistep \
        --gamma 0.1 \
        --output-dir "$BASELINE_MOBILENET_OUTPUT" \
        --save-freq 10 \
        --num-workers 4
    
    echo "✓ MobileNetV2 baseline training completed!"
fi

echo ""

# ResNet18 Baseline
echo ">>> Training ResNet18 (Baseline)..."
BASELINE_RESNET18_OUTPUT="${BASE_OUTPUT_DIR}/baseline/resnet18"

if [ -f "${BASELINE_RESNET18_OUTPUT}/checkpoints/best_model.pth" ]; then
    echo "✓ ResNet18 baseline already trained. Skipping..."
else
    python src/train_student.py \
        --train-data "$TRAIN_DATA" \
        --val-data "$TEST_DATA" \
        --num-classes $NUM_CLASSES \
        --model resnet18 \
        --distillation-mode none \
        --epochs $STUDENT_EPOCHS \
        --batch-size $STUDENT_BS \
        --lr 0.0125 \
        --optimizer sgd \
        --momentum 0.9 \
        --weight-decay 1e-4 \
        --scheduler multistep \
        --gamma 0.1 \
        --output-dir "$BASELINE_RESNET18_OUTPUT" \
        --save-freq 10 \
        --num-workers 4
    
    echo "✓ ResNet18 baseline training completed!"
fi

# ============================================================================
# STEP 3: Train Distilled Student Models
# ============================================================================
echo ""
echo "================================================================================"
echo "STEP 3: TRAINING DISTILLED STUDENT MODELS (WITH KNOWLEDGE DISTILLATION)"
echo "================================================================================"
echo ""

# MobileNetV2 Distilled
echo ">>> Training MobileNetV2 (Distilled, T=${TEMPERATURE}, α=${ALPHA})..."
DISTILLED_MOBILENET_OUTPUT="${BASE_OUTPUT_DIR}/distilled/mobilenetv2_t${TEMPERATURE}_a${ALPHA}"

if [ -f "${DISTILLED_MOBILENET_OUTPUT}/checkpoints/best_model.pth" ]; then
    echo "✓ MobileNetV2 distilled already trained. Skipping..."
else
    python src/train_student.py \
        --train-data "$TRAIN_DATA" \
        --val-data "$TEST_DATA" \
        --num-classes $NUM_CLASSES \
        --model mobilenetv2 \
        --distillation-mode online \
        --teacher-checkpoint "$TEACHER_CHECKPOINT" \
        --temperature $TEMPERATURE \
        --alpha $ALPHA \
        --epochs $STUDENT_EPOCHS \
        --batch-size $STUDENT_BS \
        --lr 0.0125 \
        --optimizer sgd \
        --momentum 0.9 \
        --weight-decay 1e-4 \
        --scheduler multistep \
        --gamma 0.1 \
        --output-dir "$DISTILLED_MOBILENET_OUTPUT" \
        --save-freq 10 \
        --num-workers 4
    
    echo "✓ MobileNetV2 distilled training completed!"
fi

echo ""

# ResNet18 Distilled
echo ">>> Training ResNet18 (Distilled, T=${TEMPERATURE}, α=${ALPHA})..."
DISTILLED_RESNET18_OUTPUT="${BASE_OUTPUT_DIR}/distilled/resnet18_t${TEMPERATURE}_a${ALPHA}"

if [ -f "${DISTILLED_RESNET18_OUTPUT}/checkpoints/best_model.pth" ]; then
    echo "✓ ResNet18 distilled already trained. Skipping..."
else
    python src/train_student.py \
        --train-data "$TRAIN_DATA" \
        --val-data "$TEST_DATA" \
        --num-classes $NUM_CLASSES \
        --model resnet18 \
        --distillation-mode online \
        --teacher-checkpoint "$TEACHER_CHECKPOINT" \
        --temperature $TEMPERATURE \
        --alpha $ALPHA \
        --epochs $STUDENT_EPOCHS \
        --batch-size $STUDENT_BS \
        --lr 0.0125 \
        --optimizer sgd \
        --momentum 0.9 \
        --weight-decay 1e-4 \
        --scheduler multistep \
        --gamma 0.1 \
        --output-dir "$DISTILLED_RESNET18_OUTPUT" \
        --save-freq 10 \
        --num-workers 4
    
    echo "✓ ResNet18 distilled training completed!"
fi

# ============================================================================
# STEP 4: Evaluate and Compare All Models
# ============================================================================
echo ""
echo "================================================================================"
echo "STEP 4: EVALUATING AND COMPARING ALL MODELS"
echo "================================================================================"
echo ""

EVAL_OUTPUT="${BASE_OUTPUT_DIR}/evaluation"

python src/evaluate_distillation.py \
    --test-data "$TEST_DATA" \
    --num-classes $NUM_CLASSES \
    --teacher-checkpoint "${TEACHER_CHECKPOINT}" \
    --baseline-checkpoints \
        "${BASELINE_MOBILENET_OUTPUT}/checkpoints/best_model.pth" \
        "${BASELINE_RESNET18_OUTPUT}/checkpoints/best_model.pth" \
    --distilled-checkpoints \
        "${DISTILLED_MOBILENET_OUTPUT}/checkpoints/best_model.pth" \
        "${DISTILLED_RESNET18_OUTPUT}/checkpoints/best_model.pth" \
    --batch-size 32 \
    --num-workers 4 \
    --output-dir "$EVAL_OUTPUT"

echo ""
echo "================================================================================"
echo "✓ KNOWLEDGE DISTILLATION PIPELINE COMPLETED!"
echo "================================================================================"
echo ""
echo "Results saved to: ${BASE_OUTPUT_DIR}/"
echo "  - Teacher: ${TEACHER_OUTPUT}"
echo "  - Baselines: ${BASE_OUTPUT_DIR}/baseline/"
echo "  - Distilled: ${BASE_OUTPUT_DIR}/distilled/"
echo "  - Evaluation: ${EVAL_OUTPUT}"
echo ""
echo "To view evaluation results:"
echo "  cat ${EVAL_OUTPUT}/evaluation_results.json"
echo ""

