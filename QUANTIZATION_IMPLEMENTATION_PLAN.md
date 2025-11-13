# 📋 QUANTIZATION IMPLEMENTATION PLAN

**Mode**: PLAN MODE  
**Objective**: Implement the missing third stage (Quantization) to complete the research punchline  
**Timeline**: 3-5 days implementation + 1 week experiments  
**Priority**: 🔴 **CRITICAL** - This is the research innovation

---

## 🎯 **Project Goal**

Complete the **Three-Stage Progressive Optimization Framework**:

```
┌─────────────────────────────────────────────────────────┐
│  Stage 1: CAPACITY MAXIMIZATION                         │
│  SE-ResNet152 (64.78M, FP32) → 99.75% accuracy         │
│  Status: ✅ COMPLETE                                    │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Stage 2: ARCHITECTURAL COMPRESSION                     │
│  MobileNetV2 (2.23M, FP32) → 99.50% accuracy           │
│  Via Knowledge Distillation                             │
│  Status: ✅ COMPLETE                                    │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Stage 3: NUMERICAL OPTIMIZATION                        │
│  MobileNetV2 (2.23M, INT8) → ~99.30% accuracy          │
│  Via Post-Training Quantization + QAT                   │
│  Status: ❌ TO BE IMPLEMENTED                           │
└─────────────────────────────────────────────────────────┘

FINAL RESULT: 116x compression, <5ms inference on edge
```

---

## 📊 **Implementation Phases**

### **PHASE 1: POST-TRAINING QUANTIZATION (PTQ)** 🔴 Priority 1
**Duration**: 1-2 days  
**Goal**: Convert trained FP32 models to INT8

#### **Task 1.1: Create PTQ Infrastructure**
**File**: `src/quantization/post_training_quantization.py`

**Components**:
1. **Dynamic Quantization**
   - Quantize weights only (activations stay FP32)
   - Fast, no calibration needed
   - Good for models with many Linear layers

2. **Static Quantization**
   - Quantize both weights and activations
   - Requires calibration data (100-1000 samples)
   - Better accuracy than dynamic

3. **Quantization Utilities**
   - Model preparation
   - Calibration data loading
   - Quantized model saving/loading

**Code Structure**:
```python
class PostTrainingQuantizer:
    def __init__(self, model, backend='fbgemm')
    def quantize_dynamic(self) -> nn.Module
    def quantize_static(self, calibration_loader) -> nn.Module
    def save_quantized_model(self, path)
    def load_quantized_model(self, path) -> nn.Module
```

**Expected Outcome**:
- MobileNetV2 FP32 (8.5MB) → INT8 (~2.1MB)
- ResNet18 FP32 (42.6MB) → INT8 (~10.7MB)
- Minimal accuracy loss (<0.5%)

---

#### **Task 1.2: Create Quantization Script**
**File**: `quantize_model.py`

**Functionality**:
```bash
# Dynamic quantization (quick)
python quantize_model.py \
    --model mobilenetv2 \
    --checkpoint experiments/distilled/mobilenetv2/checkpoints/best_model.pth \
    --method dynamic \
    --output quantized_models/mobilenetv2_dynamic.pth

# Static quantization (better accuracy)
python quantize_model.py \
    --model mobilenetv2 \
    --checkpoint experiments/distilled/mobilenetv2/checkpoints/best_model.pth \
    --method static \
    --calibration-data Training_data/load_0/train.txt \
    --calibration-samples 1000 \
    --output quantized_models/mobilenetv2_static.pth
```

**Features**:
- Support for multiple models (MobileNetV2, ResNet18)
- Dynamic and static quantization
- Calibration data selection
- Automatic backend selection (fbgemm for x86, qnnpack for ARM)
- Progress reporting

---

#### **Task 1.3: Evaluate Quantized Models**
**File**: `src/evaluate_quantized.py`

**Metrics to Measure**:
1. **Accuracy Comparison**
   - FP32 baseline
   - FP32 distilled
   - INT8 dynamic quantized
   - INT8 static quantized

2. **Inference Performance**
   - Latency (ms per sample)
   - Throughput (samples/sec)
   - Memory usage
   - CPU utilization

3. **Model Size**
   - On-disk size
   - In-memory size
   - Compression ratio

**Code Structure**:
```python
class QuantizedModelEvaluator:
    def __init__(self, test_data, num_classes=4)
    def evaluate_accuracy(self, model) -> Dict
    def benchmark_inference(self, model, num_runs=100) -> Dict
    def measure_model_size(self, model) -> Dict
    def compare_models(self, fp32_model, int8_model) -> Dict
```

**Output Format**:
```json
{
  "MobileNetV2_FP32_Distilled": {
    "accuracy": 99.50,
    "model_size_mb": 8.5,
    "inference_time_ms": 68.0,
    "memory_mb": 110.0
  },
  "MobileNetV2_INT8_Static": {
    "accuracy": 99.32,
    "model_size_mb": 2.1,
    "inference_time_ms": 35.0,
    "memory_mb": 45.0,
    "compression_ratio": 4.05,
    "speedup": 1.94,
    "accuracy_degradation": 0.18
  }
}
```

---

### **PHASE 2: QUANTIZATION-AWARE TRAINING (QAT)** 🟡 Priority 2
**Duration**: 1-2 days  
**Goal**: Train student with quantization simulation for better accuracy

#### **Task 2.1: Integrate QAT into Student Training**
**File**: Modify `src/train_student.py`

**New Arguments**:
```bash
--quantization-aware         # Enable QAT
--qat-backend {fbgemm,qnnpack}  # Backend selection
--qat-num-epochs N          # Number of QAT fine-tuning epochs
```

**Training Flow**:
```python
# 1. Standard distillation training (FP32)
for epoch in range(distillation_epochs):
    train_with_kd_loss(student_fp32, teacher, ...)

# 2. QAT fine-tuning
student_qat = prepare_qat(student_fp32)
for epoch in range(qat_epochs):
    train_with_kd_loss(student_qat, teacher, ...)

# 3. Convert to quantized
student_int8 = convert_to_quantized(student_qat)
```

**Expected Improvement**:
- QAT accuracy > PTQ accuracy by 0.1-0.3%
- Student learns to be robust to quantization noise

---

#### **Task 2.2: QAT Training Script**
**File**: `train_student_qat.py` (or integrate into existing)

**Example Usage**:
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
    --qat-backend fbgemm \
    --qat-num-epochs 20 \
    --output-dir experiments/qat/mobilenetv2_load0
```

**Key Features**:
- Seamless integration with existing KD training
- QAT as optional fine-tuning stage
- Automatic conversion to INT8 after training
- Save both FP32 and INT8 checkpoints

---

### **PHASE 3: COMPREHENSIVE BENCHMARKING** 🟡 Priority 3
**Duration**: 2-3 days  
**Goal**: Complete performance analysis across all stages

#### **Task 3.1: Create Unified Benchmark Script**
**File**: `benchmark_all_models.py`

**What It Does**:
1. Load all model variants:
   - SE-ResNet152 (Teacher, FP32)
   - MobileNetV2 Baseline (FP32)
   - MobileNetV2 Distilled (FP32)
   - MobileNetV2 PTQ-Dynamic (INT8)
   - MobileNetV2 PTQ-Static (INT8)
   - MobileNetV2 QAT (INT8)
   - ResNet18 variants (same stages)

2. Run comprehensive evaluation:
   - Accuracy on all test sets (load_0/1/2/3, Cross_load)
   - Inference time (CPU, GPU if available)
   - Memory footprint
   - Model size

3. Generate comparison tables and plots

**Output**:
- CSV file with all metrics
- Comparison plots (Pareto frontier)
- Statistical analysis (mean, std across runs)

---

#### **Task 3.2: Pareto Frontier Visualization**
**File**: `src/utils/visualizations.py` (extend)

**New Function**: `plot_efficiency_frontier()`

**Plots**:
1. **Accuracy vs Model Size**
   - X-axis: Model size (MB, log scale)
   - Y-axis: Accuracy (%)
   - Points: All model variants
   - Highlight: Pareto-optimal models

2. **Accuracy vs Inference Time**
   - X-axis: Inference time (ms, log scale)
   - Y-axis: Accuracy (%)
   - Show trade-offs

3. **3D Plot: Accuracy vs Size vs Speed**
   - X: Model size
   - Y: Inference time
   - Z: Accuracy
   - Color: Compression ratio

**Expected Insight**:
- MobileNetV2 (INT8, QAT) is Pareto-optimal
- Best accuracy/efficiency trade-off for edge deployment

---

### **PHASE 4: EDGE DEVICE DEPLOYMENT** 🟢 Priority 4 (Optional)
**Duration**: 2-3 days  
**Goal**: Real-world validation on industrial hardware

#### **Task 4.1: Raspberry Pi 4 Deployment**
**File**: `deploy_to_edge.py`

**Target Hardware**:
- Raspberry Pi 4 (ARM Cortex-A72, 4GB RAM)
- Operating Temperature: 0-50°C (industrial environment)

**Benchmark Metrics**:
1. **Latency**: End-to-end inference time
2. **Throughput**: Samples per second
3. **Power**: Watts consumed
4. **Temperature**: CPU temperature under load
5. **Real-time capability**: Can achieve >20 FPS?

**Test Scenarios**:
1. Single inference
2. Batch inference (varying batch sizes)
3. Continuous streaming (simulated sensor data)
4. Under thermal throttling conditions

---

#### **Task 4.2: NVIDIA Jetson Nano Deployment**
**File**: Same as above, different platform

**Target Hardware**:
- Jetson Nano (ARM + 128-core Maxwell GPU)
- Use TensorRT for optimization

**Additional Optimizations**:
- TensorRT INT8 optimization
- CUDA acceleration
- Mixed precision (FP16 where beneficial)

---

#### **Task 4.3: Edge Deployment Guide**
**File**: `EDGE_DEPLOYMENT_GUIDE.md`

**Contents**:
1. Hardware requirements
2. Setup instructions (OS, dependencies)
3. Model export (PyTorch → ONNX → TFLite)
4. Deployment scripts
5. Troubleshooting
6. Performance tuning tips

---

### **PHASE 5: ABLATION STUDIES & ANALYSIS** 🟢 Priority 5
**Duration**: 2-3 days  
**Goal**: Understand each component's contribution

#### **Task 5.1: Ablation Experiments**

**Experiment Matrix**:
| ID | Teacher | KD | Quantization | Expected Acc | Purpose |
|----|---------|----|--------------| -------------|---------|
| E1 | No | No | No | ~97.8% | Baseline student |
| E2 | Yes | Yes (FP32) | No | ~99.5% | KD benefit |
| E3 | Yes | Yes (FP32) | PTQ | ~99.3% | PTQ impact |
| E4 | Yes | Yes (FP32) | QAT | ~99.4% | QAT improvement |
| E5 | No | No | PTQ | ~97.5% | Quantization without KD |
| E6 | Yes | No | PTQ | ~99.0% | Teacher init + quantize |

**Key Insights**:
- E2 vs E1: KD benefit = ~1.7%
- E3 vs E2: PTQ cost = ~0.2%
- E4 vs E3: QAT improvement = ~0.1%
- E5 vs E1: Quantization alone hurts
- E6 vs E3: KD crucial for good quantized model

---

#### **Task 5.2: Hyperparameter Sensitivity**

**Test Variables**:
1. **Calibration Data Size** (for static PTQ)
   - 100, 500, 1000, 2000 samples
   - Measure: Accuracy convergence

2. **QAT Fine-tuning Epochs**
   - 10, 20, 30, 50 epochs
   - Measure: Accuracy vs training time

3. **Quantization Backend**
   - fbgemm (x86)
   - qnnpack (ARM)
   - Measure: Performance differences

4. **Mixed Precision**
   - All INT8
   - First layer FP32, rest INT8
   - Last layer FP32, rest INT8
   - Measure: Accuracy/speed trade-off

---

### **PHASE 6: DOCUMENTATION & PAPER WRITING** 📝 Priority 6
**Duration**: 3-5 days  
**Goal**: Publication-ready documentation

#### **Task 6.1: Update Technical Documentation**

**Files to Update**:
1. **README.md** - Add quantization section
2. **KD_GUIDE.md** - Add quantization tutorial
3. **API_REFERENCE.md** - Document quantization APIs
4. **QUANTIZATION_GUIDE.md** (NEW) - Complete quantization guide

**New Sections**:
- Quantization theory and motivation
- PTQ vs QAT comparison
- Step-by-step quantization tutorial
- Edge deployment instructions
- Troubleshooting quantization issues

---

#### **Task 6.2: Create Research Manuscript**
**File**: `RESEARCH_PAPER_DRAFT.md`

**Paper Structure**:

**1. Abstract** (200 words)
- Problem: Large models impractical for industrial edge deployment
- Solution: Three-stage optimization (capacity → architecture → numeric)
- Results: 116x compression, 99.5% accuracy retention, <5ms inference

**2. Introduction** (1 page)
- Industrial fault diagnosis requirements
- Deep learning deployment challenges
- Contribution: Progressive optimization framework

**3. Related Work** (1 page)
- Fault diagnosis methods
- Knowledge distillation techniques
- Model quantization approaches
- Gap: No complete optimization pipeline

**4. Methodology** (3 pages)
- **Stage 1**: SE-ResNet152 teacher training
- **Stage 2**: Knowledge distillation (T, α optimization)
- **Stage 3**: Post-training quantization + QAT
- Loss functions: L_total = α*L_CE + (1-α)*T²*KL

**5. Experiments** (3 pages)
- **Dataset**: CWRU bearing with CWT spectrograms
- **Metrics**: Accuracy, compression, inference time
- **Ablation studies**: Each stage's contribution
- **Cross-load transfer**: Generalization analysis
- **Edge deployment**: Raspberry Pi benchmarks

**6. Results** (2 pages)
- Table 1: Model comparison (all variants)
- Table 2: Ablation study results
- Figure 1: Pareto frontier (accuracy vs efficiency)
- Figure 2: Cross-load transfer performance
- Figure 3: Edge device benchmarks

**7. Discussion** (1 page)
- Why three-stage optimization works
- Trade-offs at each stage
- Industrial deployment considerations
- Limitations and future work

**8. Conclusion** (0.5 page)
- Summary of contributions
- Impact on industrial fault diagnosis
- Enabling real-time edge deployment

---

#### **Task 6.3: Prepare Supplementary Materials**

**Supplementary File 1: Complete Experimental Results**
- All accuracy numbers (all models, all loads)
- Standard deviations (5 random seeds)
- Statistical significance tests

**Supplementary File 2: Code Repository**
- Link to GitHub repository
- Instructions to reproduce all experiments
- Pre-trained model weights

**Supplementary File 3: Edge Deployment Package**
- Quantized models (ONNX, TFLite)
- Raspberry Pi deployment scripts
- Demo video (real-time inference)

---

## 📅 **Implementation Timeline**

### **Week 1: Core Quantization (Days 1-5)**
- **Day 1**: Implement PTQ infrastructure (Task 1.1)
- **Day 2**: Create quantization script (Task 1.2)
- **Day 3**: Build quantized evaluation (Task 1.3)
- **Day 4**: Integrate QAT into training (Task 2.1)
- **Day 5**: Test and debug, run initial experiments

**Milestone**: PTQ and QAT working, initial results available

---

### **Week 2: Benchmarking & Analysis (Days 6-10)**
- **Day 6**: Comprehensive benchmarking (Task 3.1)
- **Day 7**: Pareto frontier visualization (Task 3.2)
- **Day 8**: Ablation studies (Task 5.1)
- **Day 9**: Hyperparameter sensitivity (Task 5.2)
- **Day 10**: Edge deployment prep (Task 4.1 start)

**Milestone**: Complete performance analysis, ablation results

---

### **Week 3: Deployment & Documentation (Days 11-15)**
- **Day 11**: Raspberry Pi deployment (Task 4.1)
- **Day 12**: Jetson deployment (Task 4.2)
- **Day 13**: Edge deployment guide (Task 4.3)
- **Day 14**: Documentation updates (Task 6.1)
- **Day 15**: Paper draft (Task 6.2 start)

**Milestone**: Edge deployment validated, docs updated

---

### **Week 4: Paper Writing (Days 16-20)**
- **Day 16-18**: Write research paper
- **Day 19**: Prepare figures and tables
- **Day 20**: Supplementary materials

**Milestone**: Paper draft ready for submission

---

## 🎯 **Success Criteria**

### **Technical Criteria**:
1. ✅ PTQ reduces model size by 4x with <0.5% accuracy loss
2. ✅ QAT achieves better accuracy than PTQ
3. ✅ INT8 models run 2-4x faster than FP32
4. ✅ Quantized MobileNetV2 achieves <5ms inference on RPi4
5. ✅ Complete ablation study shows each stage's contribution

### **Research Criteria**:
1. ✅ Novel three-stage optimization framework
2. ✅ 116x total compression (FP32 teacher → INT8 student)
3. ✅ >99% accuracy retention (99.75% → 99.30%)
4. ✅ Real-world edge deployment validation
5. ✅ Reproducible results with public code

### **Publication Criteria**:
1. ✅ Complete manuscript (8-10 pages)
2. ✅ Comprehensive experiments and ablations
3. ✅ Pareto frontier analysis
4. ✅ Statistical significance testing
5. ✅ Open-source code repository
6. ✅ Pre-trained model weights

---

## 📊 **Expected Results**

### **Model Performance Table**:

| Model | Precision | Params | Size | Inference | Accuracy | Compression | Speedup |
|-------|-----------|--------|------|-----------|----------|-------------|---------|
| **SE-ResNet152** | FP32 | 64.78M | 247MB | 112ms | 99.75% | 1x | 1x |
| **MobileNetV2 Baseline** | FP32 | 2.23M | 8.5MB | 68ms | 97.80% | 29x | 1.6x |
| **MobileNetV2 Distilled** | FP32 | 2.23M | 8.5MB | 68ms | 99.50% | 29x | 1.6x |
| **MobileNetV2 PTQ** | INT8 | 2.23M | 2.1MB | 35ms | 99.32% | 118x | 3.2x |
| **MobileNetV2 QAT** | INT8 | 2.23M | 2.1MB | 35ms | **99.40%** | 118x | 3.2x |

**Key Achievements**:
- **116x effective compression** (64.78M FP32 → 0.56M INT8 footprint)
- **99.5% accuracy retention** (99.75% → 99.40%)
- **3.2x speedup** on CPU
- **<5ms inference** on Raspberry Pi 4

---

### **Research Contribution**:

> "We present a three-stage progressive optimization framework for deploying deep learning models on resource-constrained industrial hardware. Our approach sequentially optimizes through:
> 1. **Capacity maximization** via large teacher models
> 2. **Architectural compression** via knowledge distillation
> 3. **Numerical optimization** via quantization-aware training
>
> Applied to bearing fault diagnosis, our framework achieves 116x compression (64.78M → 0.56M effective parameters) while retaining 99.5% of teacher accuracy. The resulting INT8 MobileNetV2 model achieves <5ms inference on Raspberry Pi 4, enabling real-time fault detection in industrial environments."

---

## 🚀 **Next Steps**

### **Immediate Actions**:
1. Create `src/quantization/` directory
2. Implement `post_training_quantization.py`
3. Create `quantize_model.py` script
4. Test on a small model first
5. Run full experiments

### **Risk Mitigation**:
- **Risk**: PyTorch quantization APIs are complex
  - **Mitigation**: Start with simple dynamic quantization
- **Risk**: Accuracy degradation higher than expected
  - **Mitigation**: Use QAT if PTQ isn't sufficient
- **Risk**: Edge deployment hardware unavailable
  - **Mitigation**: Prioritize simulation, deploy later

---

## ✅ **Deliverables**

### **Code**:
1. `src/quantization/post_training_quantization.py`
2. `src/quantization/quantization_aware_training.py`
3. `quantize_model.py`
4. `src/evaluate_quantized.py`
5. `benchmark_all_models.py`
6. Updated `src/train_student.py` (QAT integration)
7. `deploy_to_edge.py`

### **Documentation**:
1. `QUANTIZATION_GUIDE.md`
2. `EDGE_DEPLOYMENT_GUIDE.md`
3. Updated `KD_GUIDE.md`
4. Updated `API_REFERENCE.md`
5. `RESEARCH_PAPER_DRAFT.md`

### **Results**:
1. Complete performance table (all models, all metrics)
2. Pareto frontier plots
3. Ablation study results
4. Edge device benchmarks
5. Statistical analysis

---

## 🎓 **Research Impact**

### **Before Quantization** (Current):
- Good KD implementation
- Standard compression results
- Academic contribution: Moderate

### **After Quantization** (Complete):
- **Novel three-stage optimization framework**
- **Industrial deployment validated**
- **Academic contribution: High**
- **Practical impact: Significant**

**Target Journals**:
- IEEE Transactions on Industrial Informatics (Impact Factor: 11.7)
- Mechanical Systems and Signal Processing (IF: 8.4)
- IEEE Transactions on Instrumentation and Measurement (IF: 5.6)

---

## 💡 **Final Recommendation**

**PROCEED WITH QUANTIZATION IMPLEMENTATION**

This is the missing piece that transforms your work from "good technical implementation" to "novel research contribution with real-world industrial impact."

**Estimated Total Time**: 3-4 weeks  
**Estimated Impact**: High (publication in top-tier journal)  
**Risk**: Low (established techniques, clear roadmap)

---

**Ready to start Phase 1: Post-Training Quantization?** 🚀

Let's implement the punchline and complete your research contribution!

