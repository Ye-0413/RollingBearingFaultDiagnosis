"""
Evaluation Script for Knowledge Distillation

Evaluates and compares:
- Teacher model
- Baseline student models
- Distilled student models

Computes metrics:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Inference Time
- Model Complexity (Parameters, FLOPs)
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_factory import create_model, get_model_info
from src.data.dataset_loader import BearingDataset


class ModelEvaluator:
    """Evaluator for comparing models"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load test data
        self.test_loader = self._create_test_loader()
        
        # Results storage
        self.results = {}
    
    def _create_test_loader(self):
        """Create test dataloader"""
        print("\n" + "=" * 80)
        print("LOADING TEST DATA")
        print("=" * 80)
        
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_dataset = BearingDataset(
            data_path=self.args.test_data,
            transform=test_transform
        )
        
        print(f"Test samples: {len(test_dataset)}")
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        return test_loader
    
    def load_model(self, model_name, checkpoint_path):
        """Load a model from checkpoint"""
        print(f"\nLoading {model_name}...")
        
        # Create model
        model = create_model(
            model_name=model_name,
            num_classes=self.args.num_classes
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        print(f"✓ Loaded model from {checkpoint_path}")
        
        return model
    
    def measure_inference_time(self, model, num_runs=100):
        """Measure average inference time"""
        print(f"  Measuring inference time ({num_runs} runs)...")
        
        # Warmup
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Measure
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = model(dummy_input)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.time()
                times.append((end - start) * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        return avg_time, std_time
    
    def evaluate_model(self, model, model_name):
        """Evaluate a single model"""
        print("\n" + "=" * 80)
        print(f"EVALUATING {model_name.upper()}")
        print("=" * 80)
        
        model.eval()
        
        all_predictions = []
        all_labels = []
        all_logits = []
        total_time = 0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc=f'Evaluating {model_name}')
            
            for images, labels, _ in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Inference
                start = time.time()
                outputs = model(images)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                total_time += time.time() - start
                
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.append(outputs.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_logits = np.concatenate(all_logits, axis=0)
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_predictions) * 100
        precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0) * 100
        recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0) * 100
        f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0) * 100
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Classification report
        class_report = classification_report(
            all_labels, all_predictions,
            target_names=[f'Class {i}' for i in range(self.args.num_classes)],
            zero_division=0
        )
        
        # Model info
        model_info = get_model_info(model)
        
        # Inference time
        avg_inference_time, std_inference_time = self.measure_inference_time(model)
        batch_inference_time = (total_time / len(self.test_loader.dataset)) * 1000  # ms per image
        
        # Collect results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'model_info': model_info,
            'inference_time_ms': avg_inference_time,
            'inference_time_std_ms': std_inference_time,
            'batch_inference_time_ms': batch_inference_time,
            'num_test_samples': len(all_labels)
        }
        
        # Print results
        print(f"\n{'='*80}")
        print(f"{model_name.upper()} RESULTS")
        print(f"{'='*80}")
        print(f"Accuracy:   {accuracy:.2f}%")
        print(f"Precision:  {precision:.2f}%")
        print(f"Recall:     {recall:.2f}%")
        print(f"F1-Score:   {f1:.2f}%")
        print(f"\nModel Complexity:")
        print(f"  Parameters: {model_info['total_parameters']:,}")
        print(f"  Model Size: {model_info['model_size_mb']:.2f} MB")
        print(f"\nInference Performance:")
        print(f"  Avg Time (single image): {avg_inference_time:.2f} ± {std_inference_time:.2f} ms")
        print(f"  Batch Avg Time: {batch_inference_time:.2f} ms")
        print(f"\nConfusion Matrix:")
        print(cm)
        print(f"\nClassification Report:")
        print(class_report)
        
        return results
    
    def compare_models(self):
        """Compare all models and generate summary"""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 80)
        
        # Create comparison table
        comparison_data = []
        
        for model_label, result in self.results.items():
            comparison_data.append({
                'Model': model_label,
                'Accuracy (%)': f"{result['accuracy']:.2f}",
                'F1-Score (%)': f"{result['f1_score']:.2f}",
                'Parameters (M)': f"{result['model_info']['total_parameters_M']:.2f}",
                'Size (MB)': f"{result['model_info']['model_size_mb']:.2f}",
                'Inference (ms)': f"{result['inference_time_ms']:.2f}"
            })
        
        # Print table
        print("\n{:<25} {:>12} {:>12} {:>15} {:>12} {:>15}".format(
            'Model', 'Accuracy', 'F1-Score', 'Parameters (M)', 'Size (MB)', 'Inference (ms)'
        ))
        print("-" * 100)
        
        for data in comparison_data:
            print("{:<25} {:>12} {:>12} {:>15} {:>12} {:>15}".format(
                data['Model'],
                data['Accuracy (%)'],
                data['F1-Score (%)'],
                data['Parameters (M)'],
                data['Size (MB)'],
                data['Inference (ms)']
            ))
        
        # Compute compression ratios and accuracy transfer
        if 'Teacher' in self.results:
            teacher_params = self.results['Teacher']['model_info']['total_parameters']
            teacher_acc = self.results['Teacher']['accuracy']
            
            print("\n" + "=" * 80)
            print("KNOWLEDGE DISTILLATION ANALYSIS")
            print("=" * 80)
            
            for model_label, result in self.results.items():
                if model_label != 'Teacher' and 'Distilled' in model_label:
                    student_params = result['model_info']['total_parameters']
                    student_acc = result['accuracy']
                    
                    compression_ratio = teacher_params / student_params
                    acc_retained = (student_acc / teacher_acc) * 100
                    acc_gap = teacher_acc - student_acc
                    
                    # Find baseline version
                    baseline_label = model_label.replace('(Distilled)', '(Baseline)').strip()
                    if baseline_label in self.results:
                        baseline_acc = self.results[baseline_label]['accuracy']
                        acc_improvement = student_acc - baseline_acc
                        
                        print(f"\n{model_label}:")
                        print(f"  Compression Ratio: {compression_ratio:.1f}x")
                        print(f"  Accuracy Retained: {acc_retained:.2f}%")
                        print(f"  Accuracy Gap: {acc_gap:.2f}%")
                        print(f"  vs Baseline Improvement: +{acc_improvement:.2f}%")
                    else:
                        print(f"\n{model_label}:")
                        print(f"  Compression Ratio: {compression_ratio:.1f}x")
                        print(f"  Accuracy Retained: {acc_retained:.2f}%")
                        print(f"  Accuracy Gap: {acc_gap:.2f}%")
        
        return comparison_data
    
    def save_results(self):
        """Save evaluation results to JSON"""
        results_path = self.output_dir / 'evaluation_results.json'
        
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {}
        for key, value in self.results.items():
            results_serializable[key] = value.copy()
            # confusion_matrix already converted to list in evaluate_model
        
        with open(results_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\n✓ Saved results to {results_path}")
    
    def run(self):
        """Run evaluation pipeline"""
        print("\n" + "=" * 80)
        print("KNOWLEDGE DISTILLATION EVALUATION")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Test data: {self.args.test_data}")
        
        # Evaluate teacher model if provided
        if self.args.teacher_checkpoint:
            teacher_model = self.load_model('se_resnet152', self.args.teacher_checkpoint)
            self.results['Teacher'] = self.evaluate_model(teacher_model, 'Teacher (SE-ResNet152)')
        
        # Evaluate baseline students if provided
        if self.args.baseline_checkpoints:
            for checkpoint_path in self.args.baseline_checkpoints:
                # Infer model name from checkpoint path or use provided names
                if 'mobilenet' in checkpoint_path.lower():
                    model_name = 'mobilenetv2'
                    label = 'MobileNetV2 (Baseline)'
                elif 'resnet18' in checkpoint_path.lower():
                    model_name = 'resnet18'
                    label = 'ResNet18 (Baseline)'
                else:
                    print(f"Warning: Could not infer model type from {checkpoint_path}, skipping")
                    continue
                
                baseline_model = self.load_model(model_name, checkpoint_path)
                self.results[label] = self.evaluate_model(baseline_model, label)
        
        # Evaluate distilled students if provided
        if self.args.distilled_checkpoints:
            for checkpoint_path in self.args.distilled_checkpoints:
                # Infer model name from checkpoint path
                if 'mobilenet' in checkpoint_path.lower():
                    model_name = 'mobilenetv2'
                    label = 'MobileNetV2 (Distilled)'
                elif 'resnet18' in checkpoint_path.lower():
                    model_name = 'resnet18'
                    label = 'ResNet18 (Distilled)'
                else:
                    print(f"Warning: Could not infer model type from {checkpoint_path}, skipping")
                    continue
                
                distilled_model = self.load_model(model_name, checkpoint_path)
                self.results[label] = self.evaluate_model(distilled_model, label)
        
        # Compare models
        if len(self.results) > 1:
            self.compare_models()
        
        # Save results
        self.save_results()
        
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETED")
        print("=" * 80)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate and Compare Models')
    
    # Data
    parser.add_argument('--test-data', type=str, required=True,
                        help='Path to test data annotation file')
    parser.add_argument('--num-classes', type=int, default=4,
                        help='Number of classes')
    
    # Model checkpoints
    parser.add_argument('--teacher-checkpoint', type=str, default=None,
                        help='Path to teacher model checkpoint')
    parser.add_argument('--baseline-checkpoints', nargs='+', default=None,
                        help='Paths to baseline student model checkpoints')
    parser.add_argument('--distilled-checkpoints', nargs='+', default=None,
                        help='Paths to distilled student model checkpoints')
    
    # Evaluation settings
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Output
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    evaluator = ModelEvaluator(args)
    evaluator.run()

