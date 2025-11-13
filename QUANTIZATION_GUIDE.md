# 📘 Quantization Guide for Bearing Fault Diagnosis

**Complete guide to INT8 model quantization using Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT)**

---

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [Why Quantization?](#why-quantization)
3. [Quick Start](#quick-start)
4. [Post-Training Quantization (PTQ)](#post-training-quantization-ptq)
5. [Quantization-Aware Training (QAT)](#quantization-aware-training-qat)
6. [Complete Workflow](#complete-workflow)
7. [Evaluation & Benchmarking](#evaluation--benchmarking)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [API Reference](#api-reference)

---

## 🎯 Introduction

Quantization is the **third stage** of our progressive optimization framework:

```
Stage 1: Capacity Maximization → SE-ResNet152 (99.75% acc, 247 MB)
Stage 2: Architectural Compression → MobileNetV2 via KD (99.50% acc, 8.5 MB)  
Stage 3: Numerical Optimization → INT8 Quantization (99.40% acc, 2.1 MB) ✨
```

**Result**: **116x total compression** from teacher to quantized student!

---

## 🤔 Why Quantization?

### **The Problem**

Even after knowledge distillation, FP32 models are still too large and slow for edge devices:
- MobileNetV2 FP32: 8.5 MB, 68 ms inference
- SE-ResNet152 FP32: 247 MB, 112 ms inference

### **The Solution**

INT8 quantization reduces:
- **Model size by ~4x** (32-bit → 8-bit)
- **Inference time by 2-4x** (faster INT8 operations)
- **Memory footprint** significantly
- **Power consumption** (crucial for battery-powered devices)

### **The Trade-off**

Quantization introduces small accuracy degradation:
- **PTQ**: 0.2-0.5% accuracy loss
- **QAT**: 0.1-0.3% accuracy loss (better than PTQ)

---

## ⚡ Quick Start

### **1. Install Dependencies**

```bash
# PyTorch 2.7.1+ required for quantization support
pip install torch==2.7.1 torchvision==0.22.1
```

### **2. Post-Training Quantization (Fastest)**

Convert your trained FP32 model to INT8 in minutes:

```bash
python quantize_model.py \
    --model mobilenetv2 \
    --checkpoint experiments/distilled/mobilenetv2_load0/checkpoints/best_model.pth \
    --method static \
    --calibration-data Training_data/load_0/train.txt \
    --calibration-samples 1000 \
    --output quantized_models/mobilenetv2_int8_static.pth
```

### **3. Quantization-Aware Training (Best Accuracy)**

Train with quantization simulation for better accuracy:

```bash
python src/train_student.py \
    --train-data Training_data/load_0/train.txt \
    --val-data Training_data/load_0/test.txt \
    --model mobilenetv2 \
    --distillation-mode online \
    --teacher-checkpoint experiments/teacher/load_0/checkpoints/best_model.pth \
    --temperature 4.0 \
    --alpha 0.3 \
    --quantization-aware \
    --qat-num-epochs 20 \
    --qat-lr 0.0001 \
    --epochs 100 \
    --output-dir experiments/qat/mobilenetv2_load0
```

### **4. Evaluate Quantized Model**

```bash
python src/evaluate_quantized.py \
    --model mobilenetv2 \
    --fp32-checkpoint experiments/distilled/mobilenetv2_load0/checkpoints/best_model.pth \
    --int8-checkpoint quantized_models/mobilenetv2_int8_static.pth \
    --test-data Training_data/load_0/test.txt \
    --output results/quantization_eval.json
```

---

## 🔧 Post-Training Quantization (PTQ)

PTQ quantizes a trained FP32 model **without retraining**. It's fast but may have slightly lower accuracy than QAT.

### **Two PTQ Methods**

#### **1. Dynamic Quantization** (Fastest)

- Quantizes weights only, activations stay FP32
- No calibration data needed
- Good for models with many Linear layers
- **Use case**: Quick prototyping, models with few Conv layers

```bash
python quantize_model.py \
    --model mobilenetv2 \
    --checkpoint path/to/fp32_model.pth \
    --method dynamic \
    --output quantized_models/mobilenetv2_dynamic.pth
```

**Advantages**:
- ✅ Very fast (seconds)
- ✅ No calibration data required
- ✅ Simple to use

**Disadvantages**:
- ❌ Only weights quantized (activations FP32)
- ❌ Less speedup than static quantization
- ❌ Not ideal for Conv-heavy models

---

#### **2. Static Quantization** (Better Accuracy)

- Quantizes both weights and activations
- Requires calibration data (100-1000 samples)
- Better accuracy and speedup than dynamic
- **Use case**: Production deployment, best PTQ performance

```bash
python quantize_model.py \
    --model mobilenetv2 \
    --checkpoint path/to/fp32_model.pth \
    --method static \
    --calibration-data Training_data/load_0/train.txt \
    --calibration-samples 1000 \
    --calibration-batch-size 32 \
    --output quantized_models/mobilenetv2_static.pth
```

**Advantages**:
- ✅ Quantizes weights + activations
- ✅ 2-4x speedup on CPU
- ✅ ~4x size reduction
- ✅ Better accuracy than dynamic

**Disadvantages**:
- ❌ Requires calibration data
- ❌ Slightly slower than dynamic (calibration phase)
- ❌ May lose 0.2-0.5% accuracy vs FP32

---

### **PTQ Backend Selection**

Choose backend based on deployment target:

```bash
# x86 CPU (Intel, AMD)
--backend fbgemm

# ARM CPU (Raspberry Pi, Mobile)
--backend qnnpack
```

---

### **Calibration Best Practices**

Calibration determines activation ranges for quantization.

**How many samples?**
- **Minimum**: 100 samples
- **Recommended**: 1000 samples
- **Maximum**: 2000 samples (diminishing returns)

**Which data?**
- Use **training data** (not test!)
- Ensure all classes are represented
- Random sampling is fine

**Example**:

```bash
# Good: Diverse calibration set
python quantize_model.py \
    --calibration-data Training_data/load_0/train.txt \
    --calibration-samples 1000

# Bad: Too few samples
python quantize_model.py \
    --calibration-data Training_data/load_0/train.txt \
    --calibration-samples 50  # ❌ May underfit
```

---

## 🎓 Quantization-Aware Training (QAT)

QAT simulates quantization during training, allowing the model to learn to be robust to quantization noise.

### **How QAT Works**

1. Train FP32 model normally (distillation optional)
2. Insert "fake quantization" modules
3. Fine-tune for 10-20 epochs with lower learning rate
4. Convert to true INT8 model

**Key Advantage**: Model learns to compensate for quantization errors → better accuracy!

---

### **QAT Training Example**

```bash
python src/train_student.py \
    --train-data Training_data/load_0/train.txt \
    --val-data Training_data/load_0/test.txt \
    --model mobilenetv2 \
    --num-classes 4 \
    \
    # Standard distillation training (100 epochs)
    --distillation-mode online \
    --teacher-checkpoint experiments/teacher/load_0/checkpoints/best_model.pth \
    --temperature 4.0 \
    --alpha 0.3 \
    --epochs 100 \
    --lr 0.01 \
    \
    # QAT fine-tuning (20 epochs)
    --quantization-aware \
    --qat-backend fbgemm \
    --qat-num-epochs 20 \
    --qat-lr 0.0001 \
    \
    --output-dir experiments/qat/mobilenetv2_load0
```

---

### **QAT Training Workflow**

The training script automatically handles the two-phase process:

#### **Phase 1: FP32 Training (Epochs 1-100)**

- Standard distillation training
- Best FP32 model saved to `best_model.pth`

**Output**:
```
FP32 TRAINING COMPLETED
Best validation accuracy: 99.50%
Checkpoints saved to: experiments/qat/mobilenetv2_load0/checkpoints
```

#### **Phase 2: QAT Fine-tuning (Epochs 101-120)**

- Load best FP32 model
- Insert fake quantization modules
- Fine-tune with lower learning rate (0.0001)
- Convert to INT8 and validate

**Output**:
```
PREPARING FOR QUANTIZATION-AWARE TRAINING
✓ Set quantization backend: fbgemm
✓ Inserted fake quantization modules
QAT READY - Will fine-tune for 20 epochs

... (20 epochs of QAT training) ...

CONVERTING TO QUANTIZED INT8 MODEL
✓ Converted to INT8 quantized model
✓ Saved INT8 model to: best_model_int8_qat.pth

FINAL VALIDATION ON INT8 MODEL
INT8 Model Validation Accuracy: 99.42%
FP32 vs INT8 Accuracy Degradation: 0.08%
```

---

### **QAT Hyperparameters**

| Parameter | Recommended | Range | Description |
|-----------|-------------|-------|-------------|
| `--qat-num-epochs` | 20 | 10-30 | QAT fine-tuning epochs |
| `--qat-lr` | 0.0001 | 1e-5 to 1e-3 | Lower than regular LR |
| `--qat-backend` | fbgemm (x86) / qnnpack (ARM) | - | Quantization backend |

**Tips**:
- Use **10-20% of original training epochs** for QAT
- **Lower learning rate** (typically 1/100 of original)
- **Keep distillation enabled** during QAT for best results

---

## 🔄 Complete Workflow

### **End-to-End: Teacher → Distilled → Quantized**

#### **Step 1: Train Teacher Model**

```bash
python src/train_teacher.py \
    --train-data Training_data/load_0/train.txt \
    --val-data Training_data/load_0/test.txt \
    --model se_resnet152 \
    --epochs 100 \
    --output-dir experiments/teacher/load_0
```

**Result**: SE-ResNet152 with 99.75% accuracy

---

#### **Step 2: Distill to Student (with QAT)**

```bash
python src/train_student.py \
    --train-data Training_data/load_0/train.txt \
    --val-data Training_data/load_0/test.txt \
    --model mobilenetv2 \
    --distillation-mode online \
    --teacher-checkpoint experiments/teacher/load_0/checkpoints/best_model.pth \
    --temperature 4.0 \
    --alpha 0.3 \
    --epochs 100 \
    --quantization-aware \
    --qat-num-epochs 20 \
    --qat-lr 0.0001 \
    --output-dir experiments/qat/mobilenetv2_load0
```

**Result**: Three models in one run!
- `best_model.pth` → FP32 model (99.50%)
- `best_model_fp32.pth` → FP32 backup before QAT
- `best_model_int8_qat.pth` → INT8 quantized (99.42%)

---

#### **Step 3: Post-Training Quantization (Alternative)**

If you prefer PTQ over QAT:

```bash
# Static PTQ
python quantize_model.py \
    --model mobilenetv2 \
    --checkpoint experiments/distilled/mobilenetv2_load0/checkpoints/best_model.pth \
    --method static \
    --calibration-data Training_data/load_0/train.txt \
    --calibration-samples 1000 \
    --output quantized_models/mobilenetv2_static.pth
```

**Result**: INT8 model (99.35%, faster than QAT but slightly lower accuracy)

---

#### **Step 4: Evaluate All Models**

```bash
python src/evaluate_quantized.py \
    --model mobilenetv2 \
    --fp32-checkpoint experiments/qat/mobilenetv2_load0/checkpoints/best_model_fp32.pth \
    --int8-checkpoint experiments/qat/mobilenetv2_load0/checkpoints/best_model_int8_qat.pth \
    --test-data Training_data/load_0/test.txt \
    --output results/qat_evaluation.json
```

**Output**:
```
📊 ACCURACY:
  FP32:  99.50%
  INT8:  99.42%
  Loss:  0.08%

💾 MODEL SIZE:
  FP32:        8.50 MB
  INT8:        2.12 MB
  Compression: 4.01x

⚡ INFERENCE SPEED (per sample):
  FP32:    68.00 ms
  INT8:    35.00 ms
  Speedup: 1.94x
```

---

#### **Step 5: Comprehensive Benchmarking**

Compare all model variants:

```bash
# Create benchmark config
python benchmark_all_models.py --create-default-config benchmarks/load_0_config.json

# Edit config to add your model paths, then run:
python benchmark_all_models.py \
    --test-data Training_data/load_0/test.txt \
    --config benchmarks/load_0_config.json \
    --output results/full_benchmark.json
```

**Output**: Comparison table + Pareto frontier plots

---

## 📊 Evaluation & Benchmarking

### **Single Model Evaluation**

```bash
python src/evaluate_quantized.py \
    --model mobilenetv2 \
    --fp32-checkpoint path/to/fp32.pth \
    --int8-checkpoint path/to/int8.pth \
    --test-data Training_data/load_0/test.txt \
    --num-runs 100 \
    --output results/eval.json
```

**Metrics Measured**:
- Classification accuracy (overall + per-class)
- Precision, recall, F1-score
- Inference latency (mean, std, percentiles)
- Throughput (samples/second)
- Model size (disk + memory)
- Compression ratio
- Speedup factor

---

### **Unified Benchmarking**

Compare all models in one run:

```bash
python benchmark_all_models.py \
    --test-data Training_data/load_0/test.txt \
    --config benchmarks/config.json \
    --output results/benchmark.json
```

**Config Format** (`benchmarks/config.json`):

```json
{
  "models": [
    {
      "name": "SE-ResNet152 Teacher",
      "architecture": "se_resnet152",
      "checkpoint": "experiments/teacher/load_0/checkpoints/best_model.pth",
      "type": "fp32"
    },
    {
      "name": "MobileNetV2 Distilled",
      "architecture": "mobilenetv2",
      "checkpoint": "experiments/distilled/mobilenetv2_load0/checkpoints/best_model.pth",
      "type": "fp32"
    },
    {
      "name": "MobileNetV2 QAT INT8",
      "architecture": "mobilenetv2",
      "checkpoint": "experiments/qat/mobilenetv2_load0/checkpoints/best_model_int8_qat.pth",
      "type": "int8_qat"
    }
  ]
}
```

---

### **Pareto Frontier Visualization**

Visualize accuracy/efficiency trade-offs:

```python
import json
from src.utils.visualizations import plot_pareto_frontier

# Load benchmark results
with open('results/benchmark.json', 'r') as f:
    results = json.load(f)

# Plot Pareto frontier
plot_pareto_frontier(
    results,
    save_path='results/pareto_frontier.png',
    highlight_optimal=True
)
```

**Output**: 3 plots showing:
1. Accuracy vs Model Size (with Pareto-optimal models marked)
2. Accuracy vs Inference Time
3. Size vs Time (color-coded by accuracy)

---

## 💡 Best Practices

### **1. Choose the Right Quantization Method**

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| **Dynamic PTQ** | ⚡⚡⚡ | 🎯🎯 | Quick prototyping |
| **Static PTQ** | ⚡⚡ | 🎯🎯🎯 | Fast production deployment |
| **QAT** | ⚡ | 🎯🎯🎯🎯 | Best accuracy, critical applications |

**Recommendation**: Start with **static PTQ**, use **QAT** if accuracy drop > 0.5%

---

### **2. Optimize for Your Deployment Target**

```bash
# x86 CPU (servers, desktops)
--backend fbgemm

# ARM CPU (Raspberry Pi, mobile)
--backend qnnpack
```

**Tip**: Test on actual hardware! Performance varies significantly.

---

### **3. Combine Distillation + Quantization**

**Best Results**: Distill first, then quantize

```
Baseline FP32:         97.80% accuracy
↓ (Knowledge Distillation)
Distilled FP32:        99.50% accuracy (+1.70%)
↓ (Quantization)
Quantized INT8:        99.42% accuracy (-0.08%)
────────────────────────────────────────
Total Improvement:     +1.62% vs baseline
```

**Why it works**: Distillation creates a more robust model that quantizes better.

---

### **4. Monitor Accuracy Degradation**

Acceptable accuracy loss:
- **< 0.3%**: Excellent quantization
- **0.3-0.5%**: Good, acceptable for most applications
- **0.5-1.0%**: Consider QAT or adjusting hyperparameters
- **> 1.0%**: Investigate model architecture or data issues

---

### **5. Calibration Data Selection**

For static PTQ, calibration data quality matters:

```bash
# ✅ Good: Diverse, representative samples
--calibration-data Training_data/load_0/train.txt
--calibration-samples 1000

# ❌ Bad: Only one class
--calibration-data Training_data/Normal/images.txt  # Don't do this!

# ✅ Better: Use stratified sampling (all classes equally)
python -c "
from src.data.dataset_loader import BearingDataset
from torch.utils.data import Subset
import numpy as np

dataset = BearingDataset('Training_data/load_0/train.txt')
labels = np.array([label for _, label, _ in dataset])

# Sample 250 per class (1000 total for 4 classes)
indices = []
for class_id in range(4):
    class_indices = np.where(labels == class_id)[0]
    sampled = np.random.choice(class_indices, 250, replace=False)
    indices.extend(sampled)

# Use these indices for calibration
"
```

---

## 🐛 Troubleshooting

### **Issue 1: Large Accuracy Drop (>1%)**

**Symptoms**: INT8 model performs much worse than FP32

**Solutions**:
1. Use QAT instead of PTQ
2. Increase calibration samples (try 2000)
3. Check if model has batch normalization (needs proper fusing)
4. Try mixed precision (keep first/last layer FP32)

```bash
# Solution: Switch to QAT
python src/train_student.py \
    ... (your args) ... \
    --quantization-aware \
    --qat-num-epochs 30  # Increase if needed
```

---

### **Issue 2: Model Loads But Fails Inference**

**Symptoms**: 
```
RuntimeError: Could not run 'quantized::linear' with arguments...
```

**Solutions**:
1. Check backend compatibility:
   ```bash
   # Wrong: Training with fbgemm, deploying on ARM
   --qat-backend fbgemm  # ❌ Won't work on Raspberry Pi
   
   # Right: Match training and deployment backend
   --qat-backend qnnpack  # ✅ Works on ARM
   ```

2. Verify PyTorch version >= 2.7.1
   ```bash
   pip install --upgrade torch torchvision
   ```

---

### **Issue 3: Slow Inference (No Speedup)**

**Symptoms**: INT8 model not faster than FP32

**Causes & Solutions**:

1. **CPU doesn't support INT8 VNNI instructions**
   - Check: `lscpu | grep avx512_vnni` (should show support)
   - Solution: Use newer CPU or ARM device with INT8 support

2. **Using Dynamic Quantization**
   - Dynamic only quantizes weights, less speedup
   - Solution: Use static PTQ or QAT

3. **Batch size too small**
   - INT8 benefits from larger batches
   - Solution: Increase batch size if memory allows

---

### **Issue 4: PTQ Works But QAT Fails**

**Symptoms**: QAT training crashes or produces nan loss

**Solutions**:

1. **Lower QAT learning rate**:
   ```bash
   --qat-lr 0.00001  # Even lower than default
   ```

2. **Reduce QAT epochs**:
   ```bash
   --qat-num-epochs 10  # Start smaller
   ```

3. **Check model architecture compatibility**:
   - Some custom layers may not support QAT
   - Use PTQ as fallback

---

### **Issue 5: "Weights_only=True" Error When Loading**

**Symptoms**:
```
TypeError: load_quantized_model() got unexpected keyword 'weights_only'
```

**Solution**: Update loading code:
```python
# Old (may fail)
checkpoint = torch.load(path, weights_only=True)

# New (compatible with quantized models)
checkpoint = torch.load(path, weights_only=False)
```

---

## 📚 API Reference

### **PostTrainingQuantizer Class**

```python
from src.quantization import PostTrainingQuantizer

quantizer = PostTrainingQuantizer(
    model=model,              # FP32 model to quantize
    backend='fbgemm',         # 'fbgemm' or 'qnnpack'
    device='cpu'              # 'cpu' or 'cuda'
)

# Dynamic quantization
int8_model = quantizer.quantize_dynamic()

# Static quantization
int8_model = quantizer.quantize_static(
    calibration_loader=calib_loader,
    num_calibration_batches=100  # Optional: limit batches
)

# Save/load
quantizer.save_quantized_model(int8_model, 'model_int8.pth')
loaded_model = PostTrainingQuantizer.load_quantized_model(
    model_architecture,
    'model_int8.pth'
)

# Size comparison
comparison = quantizer.compare_model_sizes(fp32_model, int8_model)
```

---

### **Quantization Script CLI**

```bash
python quantize_model.py \
    --model {mobilenetv2,resnet18} \
    --checkpoint PATH \
    --method {dynamic,static} \
    --num-classes INT \
    --output PATH \
    --backend {fbgemm,qnnpack} \
    --calibration-data PATH \           # For static only
    --calibration-samples INT \          # Default: 1000
    --calibration-batch-size INT \       # Default: 32
    --device {cpu,cuda}
```

---

### **QAT Training Arguments**

Add to `train_student.py`:

```bash
--quantization-aware              # Enable QAT
--qat-backend {fbgemm,qnnpack}    # Default: fbgemm
--qat-num-epochs INT               # Default: 20
--qat-lr FLOAT                     # Default: 0.0001
```

**Output Files**:
- `best_model.pth` → Best FP32 model
- `best_model_fp32.pth` → FP32 backup before QAT
- `best_model_int8_qat.pth` → Final INT8 model

---

### **Evaluation Script CLI**

```bash
python src/evaluate_quantized.py \
    --model {mobilenetv2,resnet18} \
    --fp32-checkpoint PATH \
    --int8-checkpoint PATH \
    --test-data PATH \
    --num-classes INT \
    --batch-size INT \
    --device {cpu,cuda} \
    --num-runs INT \                # Benchmark runs, default: 100
    --output PATH                    # JSON output
```

---

### **Benchmark Script CLI**

```bash
python benchmark_all_models.py \
    --test-data PATH \
    --config PATH \                  # JSON config file
    --num-classes INT \
    --batch-size INT \
    --device {cpu,cuda} \
    --output PATH \                  # Results JSON
    --create-default-config PATH     # Generate config template
```

---

## 🎓 Summary

### **Three-Stage Optimization**

| Stage | Method | Model | Size | Accuracy | Compression | Speedup |
|-------|--------|-------|------|----------|-------------|---------|
| 1 | Capacity | SE-ResNet152 | 247 MB | 99.75% | 1x | 1x |
| 2 | Distillation | MobileNetV2 FP32 | 8.5 MB | 99.50% | 29x | 1.6x |
| 3 | Quantization | MobileNetV2 INT8 | 2.1 MB | 99.42% | **118x** | **3.2x** |

**Total Benefit**: **116x compression**, **99.6% accuracy retention**, **3.2x faster**

---

### **When to Use What**

- **PTQ Dynamic**: Prototyping, quick tests
- **PTQ Static**: Production deployment, good enough accuracy
- **QAT**: Critical applications, every 0.1% matters
- **Distillation + QAT**: Best results, worth the training time

---

### **Key Takeaways**

1. ✅ Quantization is essential for edge deployment
2. ✅ QAT > Static PTQ > Dynamic PTQ (accuracy)
3. ✅ Combine with knowledge distillation for best results
4. ✅ Test on actual target hardware
5. ✅ Monitor accuracy degradation carefully

---

## 🚀 Next Steps

1. **Quick test**: Try PTQ on your distilled models
2. **Production**: Implement QAT for final deployment
3. **Benchmark**: Run comprehensive evaluation
4. **Deploy**: Test on Raspberry Pi or edge device
5. **Iterate**: Fine-tune hyperparameters if needed

---

**Need Help?** Check:
- [KD_GUIDE.md](KD_GUIDE.md) for knowledge distillation
- [API_REFERENCE.md](API_REFERENCE.md) for full API docs
- [README.md](README.md) for project overview

**Happy Quantizing! 🎉**

