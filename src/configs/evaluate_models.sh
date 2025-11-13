#!/bin/bash
# Evaluation script comparing teacher, baseline, and distilled models

# Update these paths to your actual model checkpoints
TEACHER_CHECKPOINT="experiments/teacher/load_0/checkpoints/best_model.pth"
BASELINE_MOBILENET="experiments/baseline/mobilenetv2_load0/checkpoints/best_model.pth"
BASELINE_RESNET18="experiments/baseline/resnet18_load0/checkpoints/best_model.pth"
DISTILLED_MOBILENET="experiments/distilled/mobilenetv2_load0_t4_a03/checkpoints/best_model.pth"
DISTILLED_RESNET18="experiments/distilled/resnet18_load0_t4_a03/checkpoints/best_model.pth"

python src/evaluate_distillation.py \
    --test-data Training_data/load_0/test.txt \
    --num-classes 4 \
    --teacher-checkpoint "$TEACHER_CHECKPOINT" \
    --baseline-checkpoints "$BASELINE_MOBILENET" "$BASELINE_RESNET18" \
    --distilled-checkpoints "$DISTILLED_MOBILENET" "$DISTILLED_RESNET18" \
    --batch-size 32 \
    --num-workers 4 \
    --output-dir experiments/evaluation/load_0

