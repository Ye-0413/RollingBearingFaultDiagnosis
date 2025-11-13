"""
Quantized Model Evaluation Script

Comprehensive evaluation and benchmarking for quantized models.
Compares FP32 baseline vs INT8 quantized models across multiple metrics.

Metrics:
- Classification accuracy (overall and per-class)
- Inference latency (mean, std, percentiles)
- Model size (on-disk and in-memory)
- Throughput (samples/second)
- Memory footprint

Usage:
    # Evaluate a single quantized model
    python src/evaluate_quantized.py \
        --model mobilenetv2 \
        --fp32-checkpoint experiments/distilled/mobilenetv2_load0/checkpoints/best_model.pth \
        --int8-checkpoint quantized_models/mobilenetv2_static.pth \
        --test-data Training_data/load_0/test.txt \
        --output results/quantization_eval.json
    
    # Evaluate multiple models
    python src/evaluate_quantized.py \
        --model mobilenetv2 \
        --fp32-checkpoint experiments/distilled/mobilenetv2_load0/checkpoints/best_model.pth \
        --int8-dynamic quantized_models/mobilenetv2_dynamic.pth \
        --int8-static quantized_models/mobilenetv2_static.pth \
        --test-data Training_data/load_0/test.txt

Author: Research Team
Date: 2025-11-13
"""

import os
import sys
import argparse
import time
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset_loader import BearingDataset
from src.models.mobilenetv2 import mobilenet_v2
from src.models.resnet18_light import resnet18


class QuantizedModelEvaluator:
    """
    Comprehensive evaluator for quantized models.
    
    Args:
        test_loader: DataLoader for test data
        num_classes: Number of classes
        device: Device to run evaluation on
    """
    
    def __init__(
        self,
        test_loader: DataLoader,
        num_classes: int = 4,
        device: str = 'cpu'
    ):
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.device = device
    
    def evaluate_accuracy(self, model: nn.Module) -> Dict[str, Any]:
        """
        Evaluate classification accuracy.
        
        Returns:
            Dictionary with accuracy metrics
        """
        model.eval()
        model.to(self.device)
        
        all_preds = []
        all_labels = []
        
        print("Evaluating accuracy...")
        with torch.no_grad():
            for batch in self.test_loader:
                images, labels = batch
                images = images.to(self.device)
                
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds) * 100
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        # Per-class accuracy
        per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1) * 100
        
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': conf_matrix.tolist(),
            'per_class_accuracy': per_class_acc.tolist()
        }
        
        return results
    
    def benchmark_inference(
        self,
        model: nn.Module,
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark inference performance.
        
        Args:
            model: Model to benchmark
            num_runs: Number of inference runs
            warmup_runs: Number of warmup runs (not counted)
        
        Returns:
            Dictionary with timing statistics
        """
        model.eval()
        model.to(self.device)
        
        # Get a sample batch
        sample_batch = next(iter(self.test_loader))[0].to(self.device)
        batch_size = sample_batch.size(0)
        
        print(f"Benchmarking inference ({num_runs} runs)...")
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(sample_batch)
        
        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = model(sample_batch)
                
                # Synchronize if using CUDA
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        latencies = np.array(latencies)
        
        # Per-sample latency
        per_sample_latencies = latencies / batch_size
        
        results = {
            'batch_size': batch_size,
            'num_runs': num_runs,
            'batch_latency_ms': {
                'mean': float(np.mean(latencies)),
                'std': float(np.std(latencies)),
                'min': float(np.min(latencies)),
                'max': float(np.max(latencies)),
                'median': float(np.median(latencies)),
                'p95': float(np.percentile(latencies, 95)),
                'p99': float(np.percentile(latencies, 99))
            },
            'per_sample_latency_ms': {
                'mean': float(np.mean(per_sample_latencies)),
                'std': float(np.std(per_sample_latencies)),
                'min': float(np.min(per_sample_latencies)),
                'max': float(np.max(per_sample_latencies)),
                'median': float(np.median(per_sample_latencies))
            },
            'throughput_samples_per_sec': float(1000 / np.mean(per_sample_latencies))
        }
        
        return results
    
    def measure_model_size(self, model: nn.Module) -> Dict[str, Any]:
        """
        Measure model size on disk and in memory.
        
        Returns:
            Dictionary with size metrics
        """
        import tempfile
        
        # Save to temp file to measure disk size
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=True) as f:
            torch.save(model.state_dict(), f.name)
            disk_size_bytes = os.path.getsize(f.name)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        results = {
            'disk_size_mb': disk_size_bytes / (1024 * 1024),
            'disk_size_bytes': disk_size_bytes,
            'total_params': total_params,
            'trainable_params': trainable_params
        }
        
        return results
    
    def compare_models(
        self,
        fp32_model: nn.Module,
        int8_model: nn.Module,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Comprehensive comparison between FP32 and INT8 models.
        
        Returns:
            Dictionary with all comparison metrics
        """
        print("=" * 70)
        print(f"EVALUATING: {model_name}")
        print("=" * 70)
        
        # 1. Accuracy evaluation
        print("\n[1/4] FP32 Accuracy Evaluation")
        fp32_accuracy = self.evaluate_accuracy(fp32_model)
        
        print("[2/4] INT8 Accuracy Evaluation")
        int8_accuracy = self.evaluate_accuracy(int8_model)
        
        # 2. Performance benchmarking
        print("\n[3/4] FP32 Performance Benchmark")
        fp32_performance = self.benchmark_inference(fp32_model)
        
        print("[4/4] INT8 Performance Benchmark")
        int8_performance = self.benchmark_inference(int8_model)
        
        # 3. Model size
        fp32_size = self.measure_model_size(fp32_model)
        int8_size = self.measure_model_size(int8_model)
        
        # 4. Calculate improvements
        compression_ratio = fp32_size['disk_size_mb'] / int8_size['disk_size_mb']
        speedup = (fp32_performance['per_sample_latency_ms']['mean'] /
                   int8_performance['per_sample_latency_ms']['mean'])
        accuracy_degradation = fp32_accuracy['accuracy'] - int8_accuracy['accuracy']
        
        # Compile results
        results = {
            'model_name': model_name,
            'fp32': {
                'accuracy': fp32_accuracy,
                'performance': fp32_performance,
                'size': fp32_size
            },
            'int8': {
                'accuracy': int8_accuracy,
                'performance': int8_performance,
                'size': int8_size
            },
            'comparison': {
                'compression_ratio': float(compression_ratio),
                'speedup': float(speedup),
                'accuracy_degradation_percent': float(accuracy_degradation),
                'size_reduction_mb': float(fp32_size['disk_size_mb'] - int8_size['disk_size_mb']),
                'latency_reduction_ms': float(
                    fp32_performance['per_sample_latency_ms']['mean'] -
                    int8_performance['per_sample_latency_ms']['mean']
                )
            }
        }
        
        # Print summary
        self.print_comparison_summary(results)
        
        return results
    
    def print_comparison_summary(self, results: Dict[str, Any]):
        """Print a formatted comparison summary."""
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        
        # Accuracy
        print("\n📊 ACCURACY:")
        print(f"  FP32:  {results['fp32']['accuracy']['accuracy']:.2f}%")
        print(f"  INT8:  {results['int8']['accuracy']['accuracy']:.2f}%")
        print(f"  Loss:  {results['comparison']['accuracy_degradation_percent']:.2f}%")
        
        # Model Size
        print("\n💾 MODEL SIZE:")
        print(f"  FP32:        {results['fp32']['size']['disk_size_mb']:.2f} MB")
        print(f"  INT8:        {results['int8']['size']['disk_size_mb']:.2f} MB")
        print(f"  Compression: {results['comparison']['compression_ratio']:.2f}x")
        print(f"  Saved:       {results['comparison']['size_reduction_mb']:.2f} MB")
        
        # Inference Speed
        print("\n⚡ INFERENCE SPEED (per sample):")
        print(f"  FP32:    {results['fp32']['performance']['per_sample_latency_ms']['mean']:.2f} ms")
        print(f"  INT8:    {results['int8']['performance']['per_sample_latency_ms']['mean']:.2f} ms")
        print(f"  Speedup: {results['comparison']['speedup']:.2f}x")
        print(f"  Faster:  {results['comparison']['latency_reduction_ms']:.2f} ms")
        
        # Throughput
        print("\n🚀 THROUGHPUT:")
        print(f"  FP32: {results['fp32']['performance']['throughput_samples_per_sec']:.1f} samples/sec")
        print(f"  INT8: {results['int8']['performance']['throughput_samples_per_sec']:.1f} samples/sec")
        
        print("\n" + "=" * 70)


def load_model(model_name: str, num_classes: int, checkpoint_path: str, device: str) -> nn.Module:
    """Load a model from checkpoint."""
    print(f"Loading {model_name} from {checkpoint_path}...")
    
    # Create model
    if model_name == 'mobilenetv2':
        model = mobilenet_v2(num_classes=num_classes)
    elif model_name == 'resnet18':
        model = resnet18(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate quantized models')
    
    parser.add_argument('--model', type=str, required=True,
                        choices=['mobilenetv2', 'resnet18'],
                        help='Model architecture')
    parser.add_argument('--fp32-checkpoint', type=str, required=True,
                        help='Path to FP32 model checkpoint')
    parser.add_argument('--int8-checkpoint', type=str, required=True,
                        help='Path to INT8 quantized model checkpoint')
    parser.add_argument('--test-data', type=str, required=True,
                        help='Path to test data annotation file')
    parser.add_argument('--num-classes', type=int, default=4,
                        help='Number of classes')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on (cpu or cuda)')
    parser.add_argument('--num-runs', type=int, default=100,
                        help='Number of benchmark runs')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for results JSON file')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    print("=" * 70)
    print("QUANTIZED MODEL EVALUATION")
    print("=" * 70)
    print(f"Model:      {args.model}")
    print(f"FP32:       {args.fp32_checkpoint}")
    print(f"INT8:       {args.int8_checkpoint}")
    print(f"Test Data:  {args.test_data}")
    print(f"Device:     {args.device}")
    print("=" * 70)
    
    # Load test data
    print("\nLoading test data...")
    test_dataset = BearingDataset(annotation_file=args.test_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    print(f"✓ Loaded {len(test_dataset)} test samples")
    
    # Load models
    print("\nLoading models...")
    fp32_model = load_model(args.model, args.num_classes, args.fp32_checkpoint, args.device)
    int8_model = load_model(args.model, args.num_classes, args.int8_checkpoint, args.device)
    print("✓ Models loaded")
    
    # Create evaluator
    evaluator = QuantizedModelEvaluator(
        test_loader=test_loader,
        num_classes=args.num_classes,
        device=args.device
    )
    
    # Run comparison
    results = evaluator.compare_models(
        fp32_model=fp32_model,
        int8_model=int8_model,
        model_name=args.model
    )
    
    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {args.output}")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()

