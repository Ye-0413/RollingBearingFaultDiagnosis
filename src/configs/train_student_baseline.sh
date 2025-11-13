#!/bin/bash
# Training script for baseline student (MobileNetV2) on load_0 data

python src/train_student.py \
    --train-data Training_data/load_0/train.txt \
    --val-data Training_data/load_0/test.txt \
    --num-classes 4 \
    --model mobilenetv2 \
    --distillation-mode none \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.0125 \
    --optimizer sgd \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --scheduler multistep \
    --gamma 0.1 \
    --output-dir experiments/baseline/mobilenetv2_load0 \
    --save-freq 10 \
    --num-workers 4

