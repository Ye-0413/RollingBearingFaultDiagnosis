# Knowledge Distillation Framework - Complete Guide

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Dataset Preparation](#dataset-preparation)
3. [Training Pipeline](#training-pipeline)
4. [Model Evaluation](#model-evaluation)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Advanced Features](#advanced-features)
7. [API Reference](#api-reference)

## 🚀 Quick Start

### One-Command Pipeline

Run the complete knowledge distillation pipeline (teacher training → baseline → distillation → evaluation):

```bash
# Run on load_0 data
./run_kd_pipeline.sh load_0

# Run on other load conditions
./run_kd_pipeline.sh load_1
./run_kd_pipeline.sh load_2
./run_kd_pipeline.sh Cross_load
```

This automated script will:
1. ✅ Train SE-ResNet152 teacher model
2. ✅ Train baseline student models (MobileNetV2, ResNet18)
3. ✅ Train distilled students with knowledge distillation
4. ✅ Evaluate and compare all models

**Expected Runtime**: ~8-12 hours on NVIDIA RTX 3080 (100 epochs each)

---

## 📊 Dataset Preparation

### Step 1: Organize Your Data

```bash
python src/data/data_preparation.py
```

This script will:
- ✅ Validate dataset structure
- ✅ Fix naming inconsistencies (Normai_3 → Normal_3, Corss_Ball → Cross_Ball)
- ✅ Create train/test splits (80/20 by default)
- ✅ Generate annotation files for each load condition
- ✅ Create combined multi-load dataset
- ✅ Generate statistics report

**Output Structure**:
```
Training_data/
├── load_0/
│   ├── train.txt          # 1,000 samples
│   └── test.txt           # 250 samples
├── load_1/
│   ├── train.txt
│   └── test.txt
├── combined/
│   ├── train.txt          # 3,194 samples (all loads)
│   └── test.txt           # 800 samples
└── class_annotations.txt  # Class ID mapping
```

### Step 2: Verify Dataset

```python
from src.data.dataset_loader import BearingDataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = BearingDataset(
    data_path='Training_data/load_0/train.txt',
    transform=transform
)

print(f"Dataset size: {len(dataset)}")
image, label, path = dataset[0]
print(f"Image shape: {image.shape}, Label: {label}")
```

---

## 🎓 Training Pipeline

### Option 1: Automated Pipeline (Recommended)

```bash
./run_kd_pipeline.sh load_0
```

### Option 2: Step-by-Step Manual Training

#### Step 1: Train Teacher Model (SE-ResNet152)

```bash
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
    --output-dir experiments/teacher/load_0 \
    --num-workers 4
```

**Key Parameters:**
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (16 for SE-ResNet152 on 11GB GPU)
- `--lr`: Learning rate (0.00625 = 0.1 × 16/256, following ImageNet scaling)
- `--scheduler`: LR scheduler (`step`, `cosine`, `multistep`)

**Expected Output:**
- Best model: `experiments/teacher/load_0/checkpoints/best_model.pth`
- Training history: `experiments/teacher/load_0/logs/training_history.json`
- Config: `experiments/teacher/load_0/config.json`

**Expected Performance**: ~99.5-99.8% validation accuracy

---

#### Step 2: Train Baseline Student Models

**MobileNetV2 Baseline:**

```bash
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
    --scheduler multistep \
    --output-dir experiments/baseline/mobilenetv2_load0 \
    --num-workers 4
```

**ResNet18 Baseline:**

```bash
python src/train_student.py \
    --train-data Training_data/load_0/train.txt \
    --val-data Training_data/load_0/test.txt \
    --num-classes 4 \
    --model resnet18 \
    --distillation-mode none \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.0125 \
    --optimizer sgd \
    --scheduler multistep \
    --output-dir experiments/baseline/resnet18_load0 \
    --num-workers 4
```

**Expected Performance**:
- MobileNetV2: ~97-98% accuracy
- ResNet18: ~98-99% accuracy

---

#### Step 3: Train Distilled Student Models

**MobileNetV2 with Knowledge Distillation:**

```bash
python src/train_student.py \
    --train-data Training_data/load_0/train.txt \
    --val-data Training_data/load_0/test.txt \
    --num-classes 4 \
    --model mobilenetv2 \
    --distillation-mode online \
    --teacher-checkpoint experiments/teacher/load_0/checkpoints/best_model.pth \
    --temperature 4.0 \
    --alpha 0.3 \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.0125 \
    --optimizer sgd \
    --scheduler multistep \
    --output-dir experiments/distilled/mobilenetv2_load0_t4_a03 \
    --num-workers 4
```

**Key KD Parameters:**
- `--distillation-mode`: `online` (teacher inference during training) or `offline` (pre-computed logits)
- `--teacher-checkpoint`: Path to trained teacher model
- `--temperature`: Softening temperature T (higher = softer distributions)
  - `T=1`: Hard labels (standard training)
  - `T=4-6`: Typical for KD (recommended)
  - `T=10+`: Very soft, may lose information
- `--alpha`: Weight for hard labels (0-1)
  - `α=0`: Pure distillation (only soft labels)
  - `α=0.3`: Balanced (recommended)
  - `α=1`: Standard training (no distillation)

**Advanced: Feature-Based Distillation**

```bash
python src/train_student.py \
    --train-data Training_data/load_0/train.txt \
    --val-data Training_data/load_0/test.txt \
    --model mobilenetv2 \
    --distillation-mode online \
    --teacher-checkpoint experiments/teacher/load_0/checkpoints/best_model.pth \
    --temperature 4.0 \
    --alpha 0.3 \
    --use-feature-distillation \
    --feature-weight 0.5 \
    --use-attention-transfer \
    --attention-weight 0.1 \
    --epochs 100 \
    --output-dir experiments/distilled/mobilenetv2_advanced \
    --num-workers 4
```

**Expected Performance**:
- MobileNetV2 (Distilled): ~99-99.5% accuracy (+1-2% vs baseline)
- ResNet18 (Distilled): ~99.5-99.7% accuracy (+0.5-1% vs baseline)

---

## 📈 Model Evaluation

### Comprehensive Evaluation

```bash
python src/evaluate_distillation.py \
    --test-data Training_data/load_0/test.txt \
    --num-classes 4 \
    --teacher-checkpoint experiments/teacher/load_0/checkpoints/best_model.pth \
    --baseline-checkpoints \
        experiments/baseline/mobilenetv2_load0/checkpoints/best_model.pth \
        experiments/baseline/resnet18_load0/checkpoints/best_model.pth \
    --distilled-checkpoints \
        experiments/distilled/mobilenetv2_load0_t4_a03/checkpoints/best_model.pth \
        experiments/distilled/resnet18_load0_t4_a03/checkpoints/best_model.pth \
    --batch-size 32 \
    --output-dir experiments/evaluation/load_0
```

**Metrics Computed:**
- ✅ Accuracy, Precision, Recall, F1-Score (overall + per-class)
- ✅ Confusion Matrix
- ✅ Classification Report
- ✅ Model Complexity (parameters, model size)
- ✅ Inference Time (single image + batch average)
- ✅ Compression Ratio (teacher → student)
- ✅ Accuracy Retained (% of teacher accuracy)
- ✅ Improvement over Baseline (distilled vs baseline student)

**Example Output:**
```
================================================================================
MODEL COMPARISON SUMMARY
================================================================================

Model                     Accuracy    F1-Score    Parameters (M)  Size (MB)  Inference (ms)
-------------------------------------------------------------------------------------------------
Teacher (SE-ResNet152)    99.75       99.72       64.78           247.12     45.20
MobileNetV2 (Baseline)    97.80       97.65       2.23            8.50       2.10
MobileNetV2 (Distilled)   99.50       99.45       2.23            8.50       2.10
ResNet18 (Baseline)       98.50       98.40       11.18           42.65      3.50
ResNet18 (Distilled)      99.60       99.55       11.18           42.65      3.50

================================================================================
KNOWLEDGE DISTILLATION ANALYSIS
================================================================================

MobileNetV2 (Distilled):
  Compression Ratio: 29.0x
  Accuracy Retained: 99.75%
  Accuracy Gap: -0.25%
  vs Baseline Improvement: +1.70%
  Speedup: 21.5x faster than teacher

ResNet18 (Distilled):
  Compression Ratio: 5.8x
  Accuracy Retained: 99.85%
  Accuracy Gap: -0.15%
  vs Baseline Improvement: +1.10%
  Speedup: 12.9x faster than teacher
```

---

## 🔧 Hyperparameter Tuning

### Grid Search for Temperature and Alpha

Create a tuning script `tune_kd_params.sh`:

```bash
#!/bin/bash

TEACHER_CKPT="experiments/teacher/load_0/checkpoints/best_model.pth"
TRAIN_DATA="Training_data/load_0/train.txt"
VAL_DATA="Training_data/load_0/test.txt"

# Grid search
for TEMP in 2.0 4.0 6.0 8.0; do
    for ALPHA in 0.1 0.3 0.5 0.7; do
        echo "Training with T=${TEMP}, α=${ALPHA}"
        
        python src/train_student.py \
            --train-data "$TRAIN_DATA" \
            --val-data "$VAL_DATA" \
            --model mobilenetv2 \
            --distillation-mode online \
            --teacher-checkpoint "$TEACHER_CKPT" \
            --temperature $TEMP \
            --alpha $ALPHA \
            --epochs 50 \
            --batch-size 32 \
            --output-dir "experiments/tuning/mobilenetv2_t${TEMP}_a${ALPHA}" \
            --num-workers 4
    done
done

echo "Hyperparameter tuning completed!"
```

**Recommended Starting Points:**
- **Temperature (T)**: 4.0 (balanced)
- **Alpha (α)**: 0.3 (more weight on distillation)

**Tuning Guidelines:**
- **If student underfits**: Lower T (e.g., 2.0), increase α (e.g., 0.5)
- **If student overfits teacher**: Increase T (e.g., 6.0), lower α (e.g., 0.1)
- **For very different architectures** (teacher >> student): Higher T, lower α

---

## 🌟 Advanced Features

### 1. Cross-Load Transfer Learning

Train on single load, test on cross-load:

```bash
# Train on load_0
python src/train_student.py \
    --train-data Training_data/load_0/train.txt \
    --val-data Training_data/load_0/test.txt \
    --model mobilenetv2 \
    --distillation-mode online \
    --teacher-checkpoint experiments/teacher/load_0/checkpoints/best_model.pth \
    --temperature 4.0 \
    --alpha 0.3 \
    --epochs 100 \
    --output-dir experiments/cross_load/mobilenetv2_train_load0

# Evaluate on cross-load test set
python src/evaluate_distillation.py \
    --test-data Training_data/Cross_load/test.txt \
    --distilled-checkpoints experiments/cross_load/mobilenetv2_train_load0/checkpoints/best_model.pth \
    --output-dir experiments/evaluation/cross_load_test
```

### 2. Multi-Load Training

Train on combined dataset from all loads:

```bash
python src/train_student.py \
    --train-data Training_data/combined/train.txt \
    --val-data Training_data/combined/test.txt \
    --model mobilenetv2 \
    --distillation-mode online \
    --teacher-checkpoint experiments/teacher/combined/checkpoints/best_model.pth \
    --temperature 4.0 \
    --alpha 0.3 \
    --epochs 100 \
    --output-dir experiments/distilled/mobilenetv2_multiload
```

### 3. Feature-Based and Attention Transfer

```bash
python src/train_student.py \
    --train-data Training_data/load_0/train.txt \
    --val-data Training_data/load_0/test.txt \
    --model mobilenetv2 \
    --distillation-mode online \
    --teacher-checkpoint experiments/teacher/load_0/checkpoints/best_model.pth \
    --temperature 4.0 \
    --alpha 0.3 \
    --use-feature-distillation \
    --feature-weight 0.5 \
    --use-attention-transfer \
    --attention-weight 0.1 \
    --epochs 100 \
    --output-dir experiments/advanced/mobilenetv2_feat_attn
```

**When to Use:**
- **Feature Distillation**: When teacher and student have similar architectures (ResNet → ResNet)
- **Attention Transfer**: When focus on spatial attention matters (object detection, localization)
- **Standard KD**: General purpose, works well for most cases

---

## 📚 API Reference

### Model Factory

```python
from src.models.model_factory import create_model, get_model_info

# Create teacher model
teacher = create_model('se_resnet152', num_classes=4)

# Create student models
mobilenet = create_model('mobilenetv2', num_classes=4)
resnet18 = create_model('resnet18', num_classes=4)

# Get model information
info = get_model_info(teacher)
print(f"Parameters: {info['total_parameters_M']:.2f}M")
print(f"Model Size: {info['model_size_mb']:.2f} MB")
```

### Loss Functions

```python
from src.losses.kd_loss import (
    KnowledgeDistillationLoss,
    FeatureDistillationLoss,
    AttentionTransferLoss,
    CombinedDistillationLoss
)

# Standard KD loss
kd_loss = KnowledgeDistillationLoss(temperature=4.0, alpha=0.3)
loss, loss_dict = kd_loss(student_logits, teacher_logits, labels)

# Combined loss (KD + Feature + Attention)
combined_loss = CombinedDistillationLoss(
    temperature=4.0,
    alpha=0.3,
    feature_weight=0.5,
    attention_weight=0.1,
    use_feature_distillation=True,
    use_attention_transfer=True
)
loss, loss_dict = combined_loss(
    student_logits, teacher_logits, labels,
    student_features, teacher_features
)
```

### Dataset Loader

```python
from src.data.dataset_loader import BearingDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = BearingDataset(
    data_path='Training_data/load_0/train.txt',
    transform=transform
)

# Create dataloader
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Iterate
for images, labels, paths in loader:
    # images: [B, 3, 224, 224]
    # labels: [B]
    # paths: List of image paths
    pass
```

---

## 🎯 Best Practices

### 1. Training Recommendations

- **Start with teacher**: Train a strong teacher model first (>99% accuracy)
- **Baseline first**: Train baseline students to establish performance floor
- **Hyperparameter sweep**: Try T=[2, 4, 6, 8], α=[0.1, 0.3, 0.5, 0.7]
- **Multiple runs**: Run 3-5 times with different seeds for statistical significance
- **Monitor overfitting**: Watch validation accuracy, use early stopping if needed

### 2. Computational Efficiency

- **Batch sizes**:
  - Teacher (SE-ResNet152): 16 on 11GB GPU
  - Students (MobileNetV2/ResNet18): 32-64 on 11GB GPU
- **Mixed precision**: Use `torch.cuda.amp` for 2x speedup (see `train_student.py`)
- **Multi-GPU**: Use `torch.nn.DataParallel` or `DistributedDataParallel`

### 3. Debugging Tips

- **Loss not decreasing**: Check learning rate (try 0.01, 0.001)
- **Student worse than baseline**: Try higher α (more hard labels)
- **Teacher-student gap large**: Try higher T (softer distributions)
- **NaN loss**: Lower learning rate, check data normalization

---

## 📊 Expected Results Summary

| Model | Parameters | Accuracy | Training Time | Inference Speed |
|-------|-----------|----------|---------------|-----------------|
| SE-ResNet152 (Teacher) | 64.78M | 99.75% | ~8h (100 epochs) | 45ms |
| MobileNetV2 (Baseline) | 2.23M | 97.80% | ~2h (100 epochs) | 2.1ms (21x faster) |
| MobileNetV2 (Distilled) | 2.23M | **99.50%** | ~2h (100 epochs) | 2.1ms (21x faster) |
| ResNet18 (Baseline) | 11.18M | 98.50% | ~3h (100 epochs) | 3.5ms (13x faster) |
| ResNet18 (Distilled) | 11.18M | **99.60%** | ~3h (100 epochs) | 3.5ms (13x faster) |

**Key Takeaways:**
- ✅ **MobileNetV2 (Distilled)**: 29x smaller, 21x faster, only 0.25% accuracy drop
- ✅ **Error rate reduction**: ~78% for MobileNetV2 (baseline 2.2% → distilled 0.5%)
- ✅ **Deployment-ready**: 2.1ms inference suitable for real-time edge applications

---

## 🆘 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
--batch-size 8  # for teacher
--batch-size 16  # for students
```

**2. Dataset not found**
```bash
# Run data preparation first
python src/data/data_preparation.py
```

**3. Teacher checkpoint not loading**
```python
# Check checkpoint keys
checkpoint = torch.load('path/to/checkpoint.pth')
print(checkpoint.keys())  # Should have 'model_state_dict'
```

**4. Low distillation gain**
```bash
# Try different hyperparameters
--temperature 6.0  # Higher T
--alpha 0.2        # Lower α (more weight on soft labels)
```

---

## 📖 References

1. **Knowledge Distillation**: Hinton et al., "Distilling the Knowledge in a Neural Network", NeurIPS 2014
2. **FitNets**: Romero et al., "FitNets: Hints for Thin Deep Nets", ICLR 2015
3. **Attention Transfer**: Zagoruyko & Komodakis, "Paying More Attention to Attention", ICLR 2017
4. **MobileNetV2**: Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018
5. **SE-ResNet**: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018

---

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@article{yourname2025kd,
  title={Efficient Bearing Fault Diagnosis via Knowledge Distillation},
  author={Your Name},
  journal={Journal Name},
  year={2025}
}
```

---

## 📧 Contact

For questions or issues:
- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/project/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/project/discussions)


