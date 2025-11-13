# Knowledge Distillation - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Prepare Data (One-Time Setup)
```bash
python src/data/data_preparation.py
```
✅ This creates train/test splits and annotation files for all load conditions.

### Step 2: Run KD Pipeline (Automated)
```bash
./run_kd_pipeline.sh load_0
```
✅ This trains teacher, baseline, and distilled models automatically.

### Step 3: View Results
```bash
cat experiments/load_0/evaluation/evaluation_results.json
```

**That's it!** The pipeline will take ~8-12 hours on a modern GPU.

---

## 📋 Common Commands

### Train Teacher Model
```bash
python src/train_teacher.py \
    --train-data Training_data/load_0/train.txt \
    --val-data Training_data/load_0/test.txt \
    --output-dir experiments/teacher/load_0
```

### Train Baseline Student
```bash
python src/train_student.py \
    --train-data Training_data/load_0/train.txt \
    --val-data Training_data/load_0/test.txt \
    --model mobilenetv2 \
    --distillation-mode none \
    --output-dir experiments/baseline/mobilenetv2
```

### Train Distilled Student
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

### Evaluate Models
```bash
python src/evaluate_distillation.py \
    --test-data Training_data/load_0/test.txt \
    --teacher-checkpoint experiments/teacher/load_0/checkpoints/best_model.pth \
    --baseline-checkpoints experiments/baseline/mobilenetv2/checkpoints/best_model.pth \
    --distilled-checkpoints experiments/distilled/mobilenetv2/checkpoints/best_model.pth \
    --output-dir experiments/evaluation/load_0
```

---

## 🎛️ Key Hyperparameters

### Temperature (T)
- **T=2.0**: Sharp distributions (less transfer)
- **T=4.0**: Balanced (recommended)
- **T=8.0**: Very soft distributions (more transfer)

### Alpha (α)
- **α=0.1**: Heavy distillation (90% soft labels)
- **α=0.3**: Balanced (recommended)
- **α=0.7**: More hard labels (70% ground truth)

### Learning Rate
- **Teacher**: 0.00625 (16 × 0.1 / 256)
- **Students**: 0.0125 (32 × 0.1 / 256)

---

## 📁 Output Structure

After running the pipeline, you'll have:

```
experiments/load_0/
├── teacher/
│   └── checkpoints/
│       └── best_model.pth (SE-ResNet152)
├── baseline/
│   ├── mobilenetv2/
│   │   └── checkpoints/best_model.pth
│   └── resnet18/
│       └── checkpoints/best_model.pth
├── distilled/
│   ├── mobilenetv2_t4.0_a0.3/
│   │   └── checkpoints/best_model.pth
│   └── resnet18_t4.0_a0.3/
│       └── checkpoints/best_model.pth
└── evaluation/
    └── evaluation_results.json
```

---

## 🔍 Monitor Training

Each training run creates:
- `checkpoints/` - Saved model weights
- `logs/training_history.json` - Loss and accuracy per epoch
- `config.json` - All hyperparameters used

View training progress:
```bash
tail -f experiments/teacher/load_0/logs/training_history.json
```

---

## 💡 Tips

1. **GPU Memory Issues?** Reduce batch size:
   ```bash
   --batch-size 8  # for teacher
   --batch-size 16  # for students
   ```

2. **Want faster experiments?** Use fewer epochs:
   ```bash
   --epochs 50  # instead of 100
   ```

3. **Try different models:**
   ```bash
   --model mobilenetv2  # Smallest (2.2M params)
   --model resnet18     # Larger (11M params)
   ```

4. **Test on different loads:**
   ```bash
   ./run_kd_pipeline.sh load_1
   ./run_kd_pipeline.sh Cross_load
   ```

---

## 📖 Full Documentation

- **Complete Guide**: See `KD_GUIDE.md` (50+ pages)
- **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`
- **Project Overview**: See `README.md`

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Dataset not found" | Run `python src/data/data_preparation.py` |
| "CUDA out of memory" | Reduce `--batch-size` |
| "No module named 'src'" | Run from project root directory |
| Low accuracy | Try higher epochs or different T/α |

---

## ✅ Expected Results

| Model | Accuracy | Speedup |
|-------|----------|---------|
| Teacher (SE-ResNet152) | ~99.75% | 1.0x |
| MobileNetV2 (Baseline) | ~97.8% | 21x |
| **MobileNetV2 (Distilled)** | **~99.5%** | **21x** |

**Distillation gives you ~1.7% accuracy boost for free!**

---

## 🎯 Next Steps

1. ✅ Run `python src/data/data_preparation.py`
2. ✅ Run `./run_kd_pipeline.sh load_0`
3. ✅ Check results in `experiments/load_0/evaluation/`
4. 🔬 Experiment with different hyperparameters
5. 📊 Publish your findings!

**Happy Distilling! 🚀**

