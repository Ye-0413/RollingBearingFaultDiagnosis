#!/bin/bash
# Training script for distilled student (MobileNetV2) on load_0 data with KD

# Update TEACHER_CHECKPOINT path to point to your trained teacher model
TEACHER_CHECKPOINT="experiments/teacher/load_0/checkpoints/best_model.pth"

python src/train_student.py \
    --train-data Training_data/load_0/train.txt \
    --val-data Training_data/load_0/test.txt \
    --num-classes 4 \
    --model mobilenetv2 \
    --distillation-mode online \
    --teacher-checkpoint "$TEACHER_CHECKPOINT" \
    --temperature 4.0 \
    --alpha 0.3 \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.0125 \
    --optimizer sgd \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --scheduler multistep \
    --gamma 0.1 \
    --output-dir experiments/distilled/mobilenetv2_load0_t4_a03 \
    --save-freq 10 \
    --num-workers 4

