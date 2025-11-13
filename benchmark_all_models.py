"""
Unified Model Benchmarking Script

Comprehensive benchmarking across all model variants:
- Teacher models (SE-ResNet152, FP32)
- Baseline student models (MobileNetV2/ResNet18, FP32)
- Distilled student models (MobileNetV2/ResNet18, FP32)
- PTQ quantized models (MobileNetV2/ResNet18, INT8 Dynamic/Static)
- QAT models (MobileNetV2/ResNet18, INT8)

Generates comprehensive comparison tables and visualizations.

Usage:
    python benchmark_all_models.py \
        --test-data Training_data/load_0/test.txt \
        --config benchmarks/load_0_config.json \
        --output results/benchmark_load0.json

Author: Research Team
Date: 2025-11-13
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset_loader import BearingDataset
from src.models.mobilenetv2 import mobilenet_v2
from src.models.resnet18_light import resnet18
from src.models.model_factory import create_model
from src.evaluate_quantized import QuantizedModelEvaluator


class UnifiedBenchmark:
    """
    Unified benchmarking system for all model variants.
    """
    
    def __init__(
        self,
        test_data_path: str,
        num_classes: int = 4,
        batch_size: int = 32,
        device: str = 'cpu'
    ):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device
        
        # Load test data
        print("Loading test data...")
        test_dataset = BearingDataset(annotation_file=test_data_path)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=(device == 'cuda')
        )
        print(f"✓ Loaded {len(test_dataset)} test samples")
        
        # Create evaluator
        self.evaluator = QuantizedModelEvaluator(
            self.test_loader,
            num_classes=num_classes,
            device=device
        )
        
        # Results storage
        self.results = {}
    
    def load_model(self, model_config: Dict[str, Any]):
        """
        Load a model based on configuration.
        
        Args:
            model_config: Dictionary with keys:
                - name: Model display name
                - architecture: mobilenetv2, resnet18, se_resnet152
                - checkpoint: Path to checkpoint
                - type: fp32, int8_dynamic, int8_static, int8_qat
        
        Returns:
            Loaded model
        """
        arch = model_config['architecture']
        checkpoint_path = model_config['checkpoint']
        model_type = model_config.get('type', 'fp32')
        
        print(f"\nLoading: {model_config['name']}")
        print(f"  Architecture: {arch}")
        print(f"  Type: {model_type}")
        print(f"  Checkpoint: {checkpoint_path}")
        
        # Create model
        if arch == 'mobilenetv2':
            model = mobilenet_v2(num_classes=self.num_classes)
        elif arch == 'resnet18':
            model = resnet18(num_classes=self.num_classes)
        elif arch == 'se_resnet152':
            model = create_model('se_resnet152', num_classes=self.num_classes)
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        print("  ✓ Loaded successfully")
        
        return model
    
    def benchmark_model(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run comprehensive benchmark on a single model.
        
        Returns:
            Dictionary with all metrics
        """
        model = self.load_model(model_config)
        
        # Run evaluations
        print(f"\n[1/3] Accuracy evaluation...")
        accuracy_results = self.evaluator.evaluate_accuracy(model)
        
        print(f"[2/3] Performance benchmarking...")
        performance_results = self.evaluator.benchmark_inference(model, num_runs=100)
        
        print(f"[3/3] Model size measurement...")
        size_results = self.evaluator.measure_model_size(model)
        
        # Compile results
        results = {
            'name': model_config['name'],
            'architecture': model_config['architecture'],
            'type': model_config.get('type', 'fp32'),
            'accuracy': accuracy_results,
            'performance': performance_results,
            'size': size_results
        }
        
        # Print summary
        print(f"\n✓ {model_config['name']} Results:")
        print(f"    Accuracy: {accuracy_results['accuracy']:.2f}%")
        print(f"    Model Size: {size_results['disk_size_mb']:.2f} MB")
        print(f"    Inference Time: {performance_results['per_sample_latency_ms']['mean']:.2f} ms")
        print(f"    Throughput: {performance_results['throughput_samples_per_sec']:.1f} samples/sec")
        
        return results
    
    def benchmark_all(self, model_configs: List[Dict[str, Any]]):
        """
        Benchmark all models specified in configuration.
        
        Args:
            model_configs: List of model configuration dictionaries
        """
        print("\n" + "=" * 80)
        print("UNIFIED MODEL BENCHMARKING")
        print("=" * 80)
        print(f"Total models to benchmark: {len(model_configs)}")
        print("=" * 80)
        
        for idx, config in enumerate(model_configs, 1):
            print(f"\n{'=' * 80}")
            print(f"MODEL {idx}/{len(model_configs)}")
            print(f"{'=' * 80}")
            
            try:
                results = self.benchmark_model(config)
                self.results[config['name']] = results
            except Exception as e:
                print(f"✗ Failed to benchmark {config['name']}: {e}")
                continue
        
        print("\n" + "=" * 80)
        print("BENCHMARKING COMPLETE")
        print("=" * 80)
    
    def generate_comparison_table(self) -> str:
        """
        Generate a formatted comparison table.
        
        Returns:
            Formatted table as string
        """
        if not self.results:
            return "No results available"
        
        # Table header
        table = "\n" + "=" * 120 + "\n"
        table += "MODEL COMPARISON TABLE\n"
        table += "=" * 120 + "\n"
        table += f"{'Model':<30} {'Type':<12} {'Acc %':<8} {'Size MB':<10} {'Latency ms':<12} {'Throughput':<12} {'Params M':<10}\n"
        table += "-" * 120 + "\n"
        
        # Sort by model type and architecture
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: (x[1]['architecture'], x[1]['type'])
        )
        
        # Add rows
        for name, result in sorted_results:
            acc = result['accuracy']['accuracy']
            size = result['size']['disk_size_mb']
            latency = result['performance']['per_sample_latency_ms']['mean']
            throughput = result['performance']['throughput_samples_per_sec']
            params = result['size']['total_params'] / 1e6
            model_type = result['type']
            
            table += f"{name:<30} {model_type:<12} {acc:>7.2f} {size:>9.2f} {latency:>11.2f} {throughput:>11.1f} {params:>9.2f}\n"
        
        table += "=" * 120 + "\n"
        
        return table
    
    def generate_compression_analysis(self) -> str:
        """
        Generate compression ratio analysis.
        
        Returns:
            Formatted analysis as string
        """
        if not self.results:
            return "No results available"
        
        # Find baseline (FP32 teacher) for comparison
        teacher_result = None
        for name, result in self.results.items():
            if 'teacher' in name.lower() or 'se_resnet' in result['architecture']:
                teacher_result = result
                break
        
        if not teacher_result:
            return "No teacher model found for comparison"
        
        teacher_size = teacher_result['size']['disk_size_mb']
        teacher_acc = teacher_result['accuracy']['accuracy']
        teacher_latency = teacher_result['performance']['per_sample_latency_ms']['mean']
        
        analysis = "\n" + "=" * 120 + "\n"
        analysis += "COMPRESSION & SPEEDUP ANALYSIS (vs Teacher)\n"
        analysis += "=" * 120 + "\n"
        analysis += f"{'Model':<30} {'Compression':<15} {'Speedup':<12} {'Acc Loss %':<12} {'Size Saved MB':<15}\n"
        analysis += "-" * 120 + "\n"
        
        for name, result in self.results.items():
            if name == teacher_result['name'] or result == teacher_result:
                continue
            
            size = result['size']['disk_size_mb']
            acc = result['accuracy']['accuracy']
            latency = result['performance']['per_sample_latency_ms']['mean']
            
            compression = teacher_size / size if size > 0 else 0
            speedup = teacher_latency / latency if latency > 0 else 0
            acc_loss = teacher_acc - acc
            size_saved = teacher_size - size
            
            analysis += f"{name:<30} {compression:>14.2f}x {speedup:>11.2f}x {acc_loss:>11.2f} {size_saved:>14.2f}\n"
        
        analysis += "=" * 120 + "\n"
        
        return analysis
    
    def save_results(self, output_path: str):
        """Save results to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")
    
    def print_summary(self):
        """Print comprehensive summary."""
        print(self.generate_comparison_table())
        print(self.generate_compression_analysis())


def load_config(config_path: str) -> List[Dict[str, Any]]:
    """
    Load benchmark configuration from JSON file.
    
    Config format:
    {
        "models": [
            {
                "name": "SE-ResNet152 Teacher",
                "architecture": "se_resnet152",
                "checkpoint": "path/to/checkpoint.pth",
                "type": "fp32"
            },
            ...
        ]
    }
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['models']


def create_default_config(output_path: str):
    """Create a default benchmark configuration template."""
    config = {
        "models": [
            {
                "name": "SE-ResNet152 Teacher",
                "architecture": "se_resnet152",
                "checkpoint": "experiments/teacher/load_0/checkpoints/best_model.pth",
                "type": "fp32"
            },
            {
                "name": "MobileNetV2 Baseline",
                "architecture": "mobilenetv2",
                "checkpoint": "experiments/baseline/mobilenetv2_load0/checkpoints/best_model.pth",
                "type": "fp32"
            },
            {
                "name": "MobileNetV2 Distilled",
                "architecture": "mobilenetv2",
                "checkpoint": "experiments/distilled/mobilenetv2_load0/checkpoints/best_model.pth",
                "type": "fp32"
            },
            {
                "name": "MobileNetV2 PTQ-Dynamic",
                "architecture": "mobilenetv2",
                "checkpoint": "quantized_models/mobilenetv2_dynamic.pth",
                "type": "int8_dynamic"
            },
            {
                "name": "MobileNetV2 PTQ-Static",
                "architecture": "mobilenetv2",
                "checkpoint": "quantized_models/mobilenetv2_static.pth",
                "type": "int8_static"
            },
            {
                "name": "MobileNetV2 QAT",
                "architecture": "mobilenetv2",
                "checkpoint": "experiments/qat/mobilenetv2_load0/checkpoints/best_model_int8_qat.pth",
                "type": "int8_qat"
            }
        ]
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Created default config: {output_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Unified benchmarking for all model variants',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--test-data', type=str, required=True,
                        help='Path to test data annotation file')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to benchmark configuration JSON file')
    parser.add_argument('--create-default-config', type=str, default=None,
                        help='Create a default config template at specified path')
    parser.add_argument('--num-classes', type=int, default=4,
                        help='Number of classes')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on (cpu or cuda)')
    parser.add_argument('--output', type=str, default='results/benchmark_results.json',
                        help='Output path for results JSON')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Create default config if requested
    if args.create_default_config:
        create_default_config(args.create_default_config)
        return
    
    # Load configuration
    if args.config is None:
        print("ERROR: --config is required (or use --create-default-config to generate template)")
        return
    
    model_configs = load_config(args.config)
    
    # Create benchmark
    benchmark = UnifiedBenchmark(
        test_data_path=args.test_data,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Run benchmarks
    benchmark.benchmark_all(model_configs)
    
    # Print summary
    benchmark.print_summary()
    
    # Save results
    benchmark.save_results(args.output)
    
    print("\n✅ Benchmarking complete!")


if __name__ == "__main__":
    main()

