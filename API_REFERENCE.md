# Knowledge Distillation Framework - API Reference

Complete API documentation for all modules, classes, and functions.

---

## Table of Contents

1. [Data Processing](#data-processing)
2. [Models](#models)
3. [Loss Functions](#loss-functions)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Utilities](#utilities)
7. [Command-Line Interfaces](#command-line-interfaces)

---

## Data Processing

### `src/data/data_preparation.py`

Main script for organizing and preparing the dataset.

#### Functions

##### `validate_data_structure(base_dir: str) -> bool`
Validates the structure of the Training_data directory.

**Parameters:**
- `base_dir`: Path to Training_data directory

**Returns:**
- `bool`: True if structure is valid

**Example:**
```python
valid = validate_data_structure('Training_data')
if valid:
    print("Dataset structure is valid")
```

##### `fix_naming_inconsistencies(base_dir: str)`
Fixes known naming issues (Normai_3 → Normal_3, Corss_Ball → Cross_Ball).

**Parameters:**
- `base_dir`: Path to Training_data directory

##### `create_train_test_splits(base_dir: str, test_ratio: float = 0.2, random_seed: int = 42)`
Creates train/test splits and annotation files.

**Parameters:**
- `base_dir`: Path to Training_data directory
- `test_ratio`: Fraction of data for testing (default: 0.2)
- `random_seed`: Random seed for reproducibility (default: 42)

**Returns:**
- `tuple`: (train_samples, test_samples, class_to_label mapping)

---

### `src/data/dataset_loader.py`

PyTorch Dataset for loading bearing fault images.

#### Class: `BearingDataset`

```python
class BearingDataset(Dataset):
    def __init__(self, annotation_file, transform=None, teacher_logits_file=None)
```

**Parameters:**
- `annotation_file` (str): Path to annotation file (e.g., 'Training_data/load_0/train.txt')
- `transform` (callable, optional): Transform to apply to images
- `teacher_logits_file` (str, optional): Path to pre-computed teacher logits for offline distillation

**Methods:**

##### `__len__() -> int`
Returns the number of samples in the dataset.

##### `__getitem__(idx: int) -> Tuple[Tensor, int, str]`
Returns a sample from the dataset.

**Returns:**
- `image` (Tensor): Transformed image [C, H, W]
- `label` (int): Class label
- `path` (str): Image file path

**Example:**
```python
from src.data.dataset_loader import BearingDataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

dataset = BearingDataset(
    annotation_file='Training_data/load_0/train.txt',
    transform=transform
)

image, label, path = dataset[0]
print(f"Image shape: {image.shape}, Label: {label}")
```

---

## Models

### `src/models/model_factory.py`

Unified interface for creating models.

#### Functions

##### `create_model(model_name: str, num_classes: int = 4, pretrained: bool = False, pretrained_path: str = None) -> nn.Module`

Factory function to create models.

**Parameters:**
- `model_name` (str): Model name ('se_resnet152', 'mobilenetv2', 'resnet18')
- `num_classes` (int): Number of output classes (default: 4)
- `pretrained` (bool): Use pretrained weights (default: False)
- `pretrained_path` (str): Path to checkpoint file (optional)

**Returns:**
- `model` (nn.Module): PyTorch model

**Example:**
```python
from src.models.model_factory import create_model

# Create teacher model
teacher = create_model('se_resnet152', num_classes=4)

# Create student model
student = create_model('mobilenetv2', num_classes=4)

# Create with pretrained weights
model = create_model('resnet18', num_classes=4, 
                    pretrained_path='checkpoints/best_model.pth')
```

##### `get_model_info(model: nn.Module) -> Dict`

Get detailed information about a model.

**Parameters:**
- `model` (nn.Module): PyTorch model

**Returns:**
- `info` (dict): Dictionary with keys:
  - `total_parameters` (int)
  - `trainable_parameters` (int)
  - `total_parameters_M` (float)
  - `trainable_parameters_M` (float)
  - `model_size_mb` (float)

**Example:**
```python
from src.models.model_factory import create_model, get_model_info

model = create_model('mobilenetv2', num_classes=4)
info = get_model_info(model)

print(f"Parameters: {info['total_parameters_M']:.2f}M")
print(f"Model Size: {info['model_size_mb']:.2f} MB")
```

---

### Model Classes

All model classes support:

##### `forward(x: Tensor, return_features: bool = False)`

**Parameters:**
- `x` (Tensor): Input tensor [B, C, H, W]
- `return_features` (bool): Return intermediate features (default: False)

**Returns:**
- If `return_features=False`: `logits` (Tensor) [B, num_classes]
- If `return_features=True`: `(logits, features)` where `features` is a dict

**Example:**
```python
import torch
from src.models.model_factory import create_model

model = create_model('mobilenetv2', num_classes=4)
x = torch.randn(4, 3, 224, 224)

# Standard forward pass
logits = model(x)
print(f"Logits shape: {logits.shape}")  # [4, 4]

# With feature extraction
logits, features = model(x, return_features=True)
print(f"Features keys: {features.keys()}")
print(f"Final features shape: {features['final'].shape}")
```

---

## Loss Functions

### `src/losses/kd_loss.py`

#### Class: `KnowledgeDistillationLoss`

Standard KD loss with temperature scaling and alpha weighting.

```python
class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature: float = 4.0, alpha: float = 0.3)
```

**Parameters:**
- `temperature` (float): Temperature for softening distributions (default: 4.0)
- `alpha` (float): Weight for hard labels (default: 0.3)
  - `1-alpha` weight for soft labels

**Formula:**
```
L_KD = α * L_CE(y, student_logits) + (1-α) * T² * KL(teacher_soft || student_soft)
```

**Methods:**

##### `forward(student_logits: Tensor, teacher_logits: Tensor, labels: Tensor) -> Tuple[Tensor, Dict]`

**Parameters:**
- `student_logits` (Tensor): Student model outputs [B, num_classes]
- `teacher_logits` (Tensor): Teacher model outputs [B, num_classes]
- `labels` (Tensor): Ground truth labels [B]

**Returns:**
- `loss` (Tensor): Total KD loss
- `loss_dict` (dict): Dictionary with loss components

**Example:**
```python
from src.losses.kd_loss import KnowledgeDistillationLoss
import torch

kd_loss_fn = KnowledgeDistillationLoss(temperature=4.0, alpha=0.3)

student_logits = torch.randn(8, 4)
teacher_logits = torch.randn(8, 4)
labels = torch.randint(0, 4, (8,))

loss, loss_dict = kd_loss_fn(student_logits, teacher_logits, labels)

print(f"Total Loss: {loss.item():.4f}")
print(f"Hard Loss: {loss_dict['hard_loss']:.4f}")
print(f"Soft Loss: {loss_dict['soft_loss']:.4f}")
```

---

#### Class: `FeatureDistillationLoss`

Feature-based distillation using L2 matching.

```python
class FeatureDistillationLoss(nn.Module):
    def __init__(self, feature_weight: float = 1.0, normalize: bool = True)
```

**Parameters:**
- `feature_weight` (float): Weight for feature loss (default: 1.0)
- `normalize` (bool): Normalize features before matching (default: True)

**Methods:**

##### `forward(student_features: Tensor, teacher_features: Tensor) -> Tensor`

**Parameters:**
- `student_features` (Tensor): Student features [B, C, H, W] or [B, C]
- `teacher_features` (Tensor): Teacher features [B, C, H, W] or [B, C]

**Returns:**
- `loss` (Tensor): Feature matching loss

---

#### Class: `AttentionTransferLoss`

Attention transfer loss for spatial importance matching.

```python
class AttentionTransferLoss(nn.Module):
    def __init__(self, beta: float = 1000.0)
```

**Parameters:**
- `beta` (float): Weight for attention loss (default: 1000.0)

**Methods:**

##### `forward(student_feature_maps: List[Tensor], teacher_feature_maps: List[Tensor]) -> Tensor`

**Parameters:**
- `student_feature_maps` (list): List of student spatial features
- `teacher_feature_maps` (list): List of teacher spatial features

**Returns:**
- `loss` (Tensor): Attention transfer loss

---

#### Class: `CombinedDistillationLoss`

Combines multiple distillation objectives.

```python
class CombinedDistillationLoss(nn.Module):
    def __init__(self,
                 temperature: float = 4.0,
                 alpha: float = 0.3,
                 feature_weight: float = 0.5,
                 attention_weight: float = 0.1,
                 use_feature_distillation: bool = False,
                 use_attention_transfer: bool = False)
```

**Example:**
```python
from src.losses.kd_loss import CombinedDistillationLoss

combined_loss = CombinedDistillationLoss(
    temperature=4.0,
    alpha=0.3,
    feature_weight=0.5,
    attention_weight=0.1,
    use_feature_distillation=True,
    use_attention_transfer=True
)

# With features
loss, loss_dict = combined_loss(
    student_logits,
    teacher_logits,
    labels,
    student_features={'final': s_feat, 'layer1': s_map},
    teacher_features={'final': t_feat, 'layer1': t_map}
)
```

---

## Training

### Teacher Training

#### Script: `src/train_teacher.py`

##### Command-Line Arguments

```bash
python src/train_teacher.py \
    --train-data TRAIN_DATA \
    --val-data VAL_DATA \
    [options]
```

**Required Arguments:**
- `--train-data`: Path to training annotation file
- `--val-data`: Path to validation annotation file
- `--output-dir`: Output directory for checkpoints

**Optional Arguments:**
- `--num-classes`: Number of classes (default: 4)
- `--epochs`: Number of epochs (default: 100)
- `--batch-size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 0.01)
- `--optimizer`: Optimizer ('sgd', 'adam') (default: 'sgd')
- `--momentum`: SGD momentum (default: 0.9)
- `--weight-decay`: Weight decay (default: 1e-4)
- `--scheduler`: LR scheduler ('step', 'cosine', 'multistep') (default: 'step')
- `--save-freq`: Save checkpoint every N epochs (default: 10)
- `--num-workers`: Data loading workers (default: 4)

**Example:**
```bash
python src/train_teacher.py \
    --train-data Training_data/load_0/train.txt \
    --val-data Training_data/load_0/test.txt \
    --output-dir experiments/teacher/load_0 \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.00625
```

---

### Student Training

#### Script: `src/train_student.py`

##### Command-Line Arguments

```bash
python src/train_student.py \
    --train-data TRAIN_DATA \
    --val-data VAL_DATA \
    --model {mobilenetv2,resnet18} \
    [options]
```

**Required Arguments:**
- `--train-data`: Path to training annotation file
- `--val-data`: Path to validation annotation file
- `--model`: Student model ('mobilenetv2', 'resnet18')
- `--output-dir`: Output directory

**Distillation Arguments:**
- `--distillation-mode`: Mode ('none', 'online', 'offline') (default: 'none')
- `--teacher-checkpoint`: Path to teacher checkpoint (required for KD)
- `--temperature`: Temperature for KD (default: 4.0)
- `--alpha`: Alpha for hard labels (default: 0.3)
- `--use-feature-distillation`: Enable feature distillation
- `--feature-weight`: Weight for feature loss (default: 0.5)
- `--use-attention-transfer`: Enable attention transfer
- `--attention-weight`: Weight for attention loss (default: 0.1)

**Example - Baseline:**
```bash
python src/train_student.py \
    --train-data Training_data/load_0/train.txt \
    --val-data Training_data/load_0/test.txt \
    --model mobilenetv2 \
    --distillation-mode none \
    --output-dir experiments/baseline/mobilenetv2
```

**Example - Distillation:**
```bash
python src/train_student.py \
    --train-data Training_data/load_0/train.txt \
    --val-data Training_data/load_0/test.txt \
    --model mobilenetv2 \
    --distillation-mode online \
    --teacher-checkpoint experiments/teacher/load_0/checkpoints/best_model.pth \
    --temperature 4.0 \
    --alpha 0.3 \
    --output-dir experiments/distilled/mobilenetv2
```

---

## Evaluation

### `src/evaluate_distillation.py`

Comprehensive evaluation and comparison script.

##### Command-Line Arguments

```bash
python src/evaluate_distillation.py \
    --test-data TEST_DATA \
    [--teacher-checkpoint TEACHER] \
    [--baseline-checkpoints BASELINE ...] \
    [--distilled-checkpoints DISTILLED ...] \
    --output-dir OUTPUT
```

**Arguments:**
- `--test-data`: Path to test annotation file
- `--num-classes`: Number of classes (default: 4)
- `--teacher-checkpoint`: Teacher model checkpoint (optional)
- `--baseline-checkpoints`: List of baseline checkpoints (optional)
- `--distilled-checkpoints`: List of distilled checkpoints (optional)
- `--batch-size`: Batch size (default: 32)
- `--output-dir`: Output directory for results

**Output:**
- `evaluation_results.json`: Detailed metrics for all models

**Example:**
```bash
python src/evaluate_distillation.py \
    --test-data Training_data/load_0/test.txt \
    --teacher-checkpoint experiments/teacher/load_0/checkpoints/best_model.pth \
    --baseline-checkpoints experiments/baseline/mobilenetv2/checkpoints/best_model.pth \
    --distilled-checkpoints experiments/distilled/mobilenetv2/checkpoints/best_model.pth \
    --output-dir experiments/evaluation/load_0
```

---

## Utilities

### `src/utils/model_complexity.py`

Model complexity analysis tools.

#### Functions

##### `count_flops(model: nn.Module, input_size: Tuple = (1, 3, 224, 224), device: str = 'cpu') -> int`

Count total FLOPs for a model.

**Parameters:**
- `model`: PyTorch model
- `input_size`: Input tensor size (B, C, H, W)
- `device`: Device to run on

**Returns:**
- `total_flops` (int): Total FLOPs count

##### `measure_inference_time(model: nn.Module, input_size: Tuple = (1, 3, 224, 224), device: str = 'cpu', num_runs: int = 100) -> Dict`

Measure model inference time.

**Returns:**
- `timing_stats` (dict): Dictionary with timing statistics:
  - `mean_ms`, `std_ms`, `min_ms`, `max_ms`, `median_ms`, `p95_ms`, `p99_ms`, `fps`

##### `analyze_model_complexity(model: nn.Module, model_name: str = "Model", input_size: Tuple = (1, 3, 224, 224), device: str = 'cpu', verbose: bool = True) -> Dict`

Comprehensive model complexity analysis.

**Returns:**
- `analysis` (dict): Complete analysis including parameters, FLOPs, timing, memory

**Example:**
```python
from src.utils.model_complexity import analyze_model_complexity
from src.models.model_factory import create_model

model = create_model('mobilenetv2', num_classes=4)

results = analyze_model_complexity(
    model,
    model_name='MobileNetV2',
    input_size=(1, 3, 224, 224),
    device='cpu',
    verbose=True
)

print(f"Parameters: {results['parameters']['total_M']:.2f}M")
print(f"GFLOPs: {results['flops']['gflops']:.2f}")
print(f"Inference: {results['inference_time']['mean_ms']:.2f}ms")
```

---

### `src/utils/visualizations.py`

Visualization utilities.

#### Functions

##### `plot_training_curves(history_path: str, save_path: str = None, show: bool = True)`

Plot training and validation curves.

**Parameters:**
- `history_path`: Path to training_history.json
- `save_path`: Path to save figure (optional)
- `show`: Display the plot (default: True)

##### `plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str = None, show: bool = True)`

Plot confusion matrix.

##### `plot_model_comparison(results: Dict, save_path: str = None, show: bool = True)`

Plot model comparison (accuracy, size, speed).

##### `plot_hyperparameter_heatmap(results_path: str, save_path: str = None, show: bool = True)`

Plot hyperparameter tuning results as heatmap.

**Example:**
```python
from src.utils.visualizations import plot_training_curves, plot_model_comparison

# Plot training curves
plot_training_curves(
    'experiments/baseline/mobilenetv2/logs/training_history.json',
    save_path='figures/training_curves.png'
)

# Plot model comparison
results = {...}  # From evaluation
plot_model_comparison(results, save_path='figures/comparison.png')
```

---

## Command-Line Interfaces

### Full Pipeline

```bash
./run_kd_pipeline.sh [load_condition]
```

Runs complete KD pipeline:
1. Train teacher
2. Train baseline students
3. Train distilled students
4. Evaluate all models

**Example:**
```bash
./run_kd_pipeline.sh load_0
./run_kd_pipeline.sh Cross_load
```

---

### Hyperparameter Tuning

```bash
python tune_hyperparameters.py \
    --train-data TRAIN_DATA \
    --val-data VAL_DATA \
    --teacher-checkpoint TEACHER \
    --model {mobilenetv2,resnet18} \
    [--temperatures T1 T2 ...] \
    [--alphas A1 A2 ...] \
    --output-dir OUTPUT
```

**Example:**
```bash
python tune_hyperparameters.py \
    --train-data Training_data/load_0/train.txt \
    --val-data Training_data/load_0/test.txt \
    --teacher-checkpoint experiments/teacher/load_0/checkpoints/best_model.pth \
    --model mobilenetv2 \
    --temperatures 2.0 4.0 6.0 8.0 \
    --alphas 0.1 0.3 0.5 0.7 \
    --epochs 50 \
    --output-dir experiments/tuning/mobilenetv2
```

---

## Configuration Files

All configuration scripts are in `src/configs/`:

- `train_teacher_load0.sh` - Teacher training example
- `train_student_baseline.sh` - Baseline training example
- `train_student_distilled.sh` - Distillation training example
- `evaluate_models.sh` - Evaluation example

**Usage:**
```bash
chmod +x src/configs/train_teacher_load0.sh
./src/configs/train_teacher_load0.sh
```

---

## Output Formats

### Training History (`logs/training_history.json`)

```json
{
  "train": [
    {
      "epoch": 1,
      "loss": 1.234,
      "acc": 78.5,
      "loss_components": {
        "hard_loss": 1.5,
        "soft_loss": 0.9
      }
    }
  ],
  "val": [
    {
      "epoch": 1,
      "loss": 1.456,
      "acc": 75.2
    }
  ]
}
```

### Evaluation Results (`evaluation_results.json`)

```json
{
  "Teacher (SE-ResNet152)": {
    "model_name": "Teacher (SE-ResNet152)",
    "accuracy": 99.75,
    "precision": 99.72,
    "recall": 99.70,
    "f1_score": 99.71,
    "confusion_matrix": [[...]],
    "model_info": {
      "total_parameters": 64781044,
      "total_parameters_M": 64.78,
      "model_size_mb": 247.12
    },
    "inference_time_ms": 45.2,
    "num_test_samples": 230
  }
}
```

---

## Error Handling

### Common Errors

**1. CUDA Out of Memory**
```python
RuntimeError: CUDA out of memory
```
Solution: Reduce `--batch-size`

**2. Checkpoint Loading Error**
```python
RuntimeError: Failed to load checkpoint with weights_only=True
```
Solution: Checkpoint may contain unsafe objects. Verify source.

**3. Dataset Not Found**
```python
FileNotFoundError: Training_data/load_0/train.txt not found
```
Solution: Run `python src/data/data_preparation.py`

---

## Best Practices

### 1. Hyperparameter Selection
```python
# Recommended starting points
temperature = 4.0  # Balanced softness
alpha = 0.3        # More weight on distillation
lr = 0.0125        # 32 × 0.1 / 256 (ImageNet scaling)
```

### 2. Device Selection
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### 3. Reproducibility
```python
import torch
import random
import numpy as np

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
```

---

## Version Information

- **Framework Version**: 1.0.0
- **PyTorch Version**: 2.7.1+
- **Python Version**: 3.8+

---

## Support

For issues or questions:
- Check `KD_GUIDE.md` for usage examples
- Check `QUICK_START.md` for quick reference
- Review `TEST_REPORT.md` for expected behavior
- See `FINAL_SUMMARY.md` for implementation details

---

**Last Updated**: 2025-11-13

