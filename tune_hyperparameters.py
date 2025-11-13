#!/usr/bin/env python3
"""
Hyperparameter Tuning Script for Knowledge Distillation

Performs grid search over temperature and alpha parameters to find optimal KD settings.
"""

import os
import sys
import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime
import itertools


def run_experiment(
    model,
    train_data,
    val_data,
    teacher_checkpoint,
    temperature,
    alpha,
    output_dir,
    epochs=50,
    batch_size=32,
    lr=0.0125
):
    """Run a single distillation experiment"""
    
    cmd = [
        'python', 'src/train_student.py',
        '--train-data', train_data,
        '--val-data', val_data,
        '--num-classes', '4',
        '--model', model,
        '--distillation-mode', 'online',
        '--teacher-checkpoint', teacher_checkpoint,
        '--temperature', str(temperature),
        '--alpha', str(alpha),
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--lr', str(lr),
        '--optimizer', 'sgd',
        '--momentum', '0.9',
        '--weight-decay', '1e-4',
        '--scheduler', 'multistep',
        '--gamma', '0.1',
        '--output-dir', output_dir,
        '--save-freq', '10',
        '--num-workers', '4'
    ]
    
    print(f"\n{'='*80}")
    print(f"Running: T={temperature}, α={alpha}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")
        return False


def extract_best_accuracy(output_dir):
    """Extract best validation accuracy from training history"""
    history_path = Path(output_dir) / 'logs' / 'training_history.json'
    
    if not history_path.exists():
        return None
    
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        val_history = history.get('val', [])
        if not val_history:
            return None
        
        best_acc = max(entry['acc'] for entry in val_history)
        return best_acc
    
    except Exception as e:
        print(f"Error reading history: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for Knowledge Distillation')
    
    # Data
    parser.add_argument('--train-data', type=str, required=True,
                        help='Path to training data annotation file')
    parser.add_argument('--val-data', type=str, required=True,
                        help='Path to validation data annotation file')
    parser.add_argument('--teacher-checkpoint', type=str, required=True,
                        help='Path to teacher checkpoint')
    
    # Model
    parser.add_argument('--model', type=str, default='mobilenetv2',
                        choices=['mobilenetv2', 'resnet18'],
                        help='Student model to tune')
    
    # Hyperparameter ranges
    parser.add_argument('--temperatures', nargs='+', type=float,
                        default=[2.0, 4.0, 6.0, 8.0],
                        help='Temperature values to try')
    parser.add_argument('--alphas', nargs='+', type=float,
                        default=[0.1, 0.2, 0.3, 0.5, 0.7],
                        help='Alpha values to try')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs per experiment (use fewer for faster tuning)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0125,
                        help='Learning rate')
    
    # Output
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Base output directory for tuning experiments')
    
    args = parser.parse_args()
    
    # Create output directory
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config_path = output_base / 'tuning_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("=" * 80)
    print("KNOWLEDGE DISTILLATION HYPERPARAMETER TUNING")
    print("=" * 80)
    print(f"\nModel: {args.model}")
    print(f"Teacher: {args.teacher_checkpoint}")
    print(f"Temperatures: {args.temperatures}")
    print(f"Alphas: {args.alphas}")
    print(f"Total experiments: {len(args.temperatures) * len(args.alphas)}")
    print(f"Epochs per experiment: {args.epochs}")
    print(f"Output: {output_base}")
    
    # Grid search
    results = []
    total_experiments = len(args.temperatures) * len(args.alphas)
    current_experiment = 0
    
    for temperature, alpha in itertools.product(args.temperatures, args.alphas):
        current_experiment += 1
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {current_experiment}/{total_experiments}")
        print(f"Temperature: {temperature}, Alpha: {alpha}")
        print(f"{'='*80}")
        
        # Create experiment output directory
        exp_name = f"{args.model}_T{temperature}_A{alpha}"
        exp_output = output_base / exp_name
        
        # Run experiment
        success = run_experiment(
            model=args.model,
            train_data=args.train_data,
            val_data=args.val_data,
            teacher_checkpoint=args.teacher_checkpoint,
            temperature=temperature,
            alpha=alpha,
            output_dir=str(exp_output),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
        
        # Extract results
        if success:
            best_acc = extract_best_accuracy(exp_output)
            
            result = {
                'temperature': temperature,
                'alpha': alpha,
                'best_accuracy': best_acc,
                'output_dir': str(exp_output),
                'success': True
            }
        else:
            result = {
                'temperature': temperature,
                'alpha': alpha,
                'best_accuracy': None,
                'output_dir': str(exp_output),
                'success': False
            }
        
        results.append(result)
        
        # Save intermediate results
        results_path = output_base / 'tuning_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nExperiment {current_experiment}/{total_experiments} complete")
        if best_acc is not None:
            print(f"Best Accuracy: {best_acc:.2f}%")
    
    # Final summary
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("=" * 80)
    
    # Filter successful experiments
    successful_results = [r for r in results if r['success'] and r['best_accuracy'] is not None]
    
    if successful_results:
        # Sort by accuracy
        successful_results.sort(key=lambda x: x['best_accuracy'], reverse=True)
        
        print("\nTOP 5 CONFIGURATIONS:")
        print(f"{'Rank':<6} {'Temperature':<12} {'Alpha':<8} {'Accuracy':<12} {'Output Directory':<50}")
        print("-" * 100)
        
        for i, result in enumerate(successful_results[:5], 1):
            print(f"{i:<6} {result['temperature']:<12.1f} {result['alpha']:<8.2f} "
                  f"{result['best_accuracy']:<12.2f} {Path(result['output_dir']).name:<50}")
        
        # Best configuration
        best_result = successful_results[0]
        print(f"\n{'='*80}")
        print("BEST CONFIGURATION:")
        print(f"{'='*80}")
        print(f"  Temperature: {best_result['temperature']}")
        print(f"  Alpha:       {best_result['alpha']}")
        print(f"  Accuracy:    {best_result['best_accuracy']:.2f}%")
        print(f"  Checkpoint:  {best_result['output_dir']}/checkpoints/best_model.pth")
        
        # Save best configuration separately
        best_config_path = output_base / 'best_configuration.json'
        with open(best_config_path, 'w') as f:
            json.dump(best_result, f, indent=2)
        
        print(f"\n✓ Best configuration saved to: {best_config_path}")
    else:
        print("\n⚠ No successful experiments found")
    
    print(f"\n✓ All results saved to: {output_base / 'tuning_results.json'}")
    print(f"✓ Tuning configuration saved to: {config_path}")


if __name__ == '__main__':
    main()

