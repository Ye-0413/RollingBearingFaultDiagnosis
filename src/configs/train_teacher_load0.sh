#!/bin/bash
# Training script for SE-ResNet152 teacher on load_0 data

python src/train_teacher.py \
    --train-data Training_data/load_0/train.txt \
    --val-data Training_data/load_0/test.txt \
    --num-classes 4 \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.00625 \
    --optimizer sgd \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --scheduler multistep \
    --gamma 0.1 \
    --output-dir experiments/teacher/load_0 \
    --save-freq 10 \
    --num-workers 4

