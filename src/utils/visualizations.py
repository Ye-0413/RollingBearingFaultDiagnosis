"""
Visualization Utilities for Knowledge Distillation

Provides tools for visualizing:
- Training curves (loss, accuracy)
- Soft target distributions
- Feature maps comparison  
- Confusion matrices
- Hyperparameter tuning results
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_training_curves(history_path: str, save_path: Optional[str] = None, show: bool = True):
    """
    Plot training and validation curves from history JSON
    
    Args:
        history_path: Path to training_history.json
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    train_history = history.get('train', [])
    val_history = history.get('val', [])
    
    if not train_history or not val_history:
        print("No training history found")
        return
    
    epochs = [entry['epoch'] for entry in train_history]
    train_loss = [entry['loss'] for entry in train_history]
    train_acc = [entry['acc'] for entry in train_history]
    val_loss = [entry['loss'] for entry in val_history]
    val_acc = [entry['acc'] for entry in val_history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, train_loss, label='Train Loss', linewidth=2, marker='o', markersize=3)
    ax1.plot(epochs, val_loss, label='Val Loss', linewidth=2, marker='s', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_acc, label='Train Accuracy', linewidth=2, marker='o', markersize=3)
    ax2.plot(epochs, val_acc, label='Val Accuracy', linewidth=2, marker='s', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training curves to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                         save_path: Optional[str] = None, show: bool = True):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        save_path: Path to save figure
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax)
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_model_comparison(results: Dict, save_path: Optional[str] = None, show: bool = True):
    """
    Plot model comparison (accuracy, size, inference time)
    
    Args:
        results: Dictionary of evaluation results
        save_path: Path to save figure
        show: Whether to display the plot
    """
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    params_m = [results[m]['model_info']['total_parameters_M'] for m in models]
    inference_ms = [results[m]['inference_time_ms'] for m in models]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Accuracy
    colors = ['#2ecc71' if 'Teacher' in m else '#3498db' if 'Baseline' in m else '#e74c3c' for m in models]
    ax1.barh(models, accuracies, color=colors, alpha=0.7)
    ax1.set_xlabel('Accuracy (%)')
    ax1.set_title('Model Accuracy Comparison')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Parameters
    ax2.barh(models, params_m, color=colors, alpha=0.7)
    ax2.set_xlabel('Parameters (M)')
    ax2.set_title('Model Size Comparison')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Inference time
    ax3.barh(models, inference_ms, color=colors, alpha=0.7)
    ax3.set_xlabel('Inference Time (ms)')
    ax3.set_title('Inference Speed Comparison')
    ax3.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved model comparison to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_kd_loss_components(history_path: str, save_path: Optional[str] = None, show: bool = True):
    """
    Plot KD loss components (hard loss, soft loss, total loss)
    
    Args:
        history_path: Path to training_history.json
        save_path: Path to save figure
        show: Whether to display the plot
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    train_history = history.get('train', [])
    
    if not train_history:
        print("No training history found")
        return
    
    # Check if loss components are available
    if 'loss_components' not in train_history[0]:
        print("No KD loss components found in history")
        return
    
    epochs = [entry['epoch'] for entry in train_history]
    hard_losses = [entry['loss_components'].get('hard_loss', 0) for entry in train_history]
    soft_losses = [entry['loss_components'].get('soft_loss', 0) for entry in train_history]
    total_losses = [entry['loss'] for entry in train_history]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(epochs, hard_losses, label='Hard Loss (CE)', linewidth=2, marker='o', markersize=3)
    ax.plot(epochs, soft_losses, label='Soft Loss (KL)', linewidth=2, marker='s', markersize=3)
    ax.plot(epochs, total_losses, label='Total Loss', linewidth=2, marker='^', markersize=3, linestyle='--')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Knowledge Distillation Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved KD loss components to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_hyperparameter_heatmap(results_path: str, save_path: Optional[str] = None, show: bool = True):
    """
    Plot hyperparameter tuning results as a heatmap
    
    Args:
        results_path: Path to tuning_results.json
        save_path: Path to save figure
        show: Whether to display the plot
    """
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Extract temperatures, alphas, and accuracies
    successful_results = [r for r in results if r['success'] and r['best_accuracy'] is not None]
    
    if not successful_results:
        print("No successful tuning results found")
        return
    
    temperatures = sorted(set(r['temperature'] for r in successful_results))
    alphas = sorted(set(r['alpha'] for r in successful_results))
    
    # Create accuracy matrix
    acc_matrix = np.zeros((len(alphas), len(temperatures)))
    
    for result in successful_results:
        t_idx = temperatures.index(result['temperature'])
        a_idx = alphas.index(result['alpha'])
        acc_matrix[a_idx, t_idx] = result['best_accuracy']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(acc_matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
                xticklabels=[f'T={t}' for t in temperatures],
                yticklabels=[f'α={a}' for a in alphas],
                cbar_kws={'label': 'Accuracy (%)'},
                vmin=acc_matrix.min(), vmax=acc_matrix.max(),
                ax=ax)
    
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Alpha')
    ax.set_title('Hyperparameter Tuning Results')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved hyperparameter heatmap to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_accuracy_vs_complexity(results: Dict, save_path: Optional[str] = None, show: bool = True):
    """
    Plot accuracy vs model complexity (Pareto frontier)
    
    Args:
        results: Dictionary of evaluation results
        save_path: Path to save figure
        show: Whether to display the plot
    """
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    params_m = [results[m]['model_info']['total_parameters_M'] for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color code: teacher, baseline, distilled
    colors = []
    markers = []
    sizes = []
    
    for m in models:
        if 'Teacher' in m:
            colors.append('#2ecc71')
            markers.append('*')
            sizes.append(300)
        elif 'Baseline' in m:
            colors.append('#3498db')
            markers.append('o')
            sizes.append(150)
        else:  # Distilled
            colors.append('#e74c3c')
            markers.append('D')
            sizes.append(150)
    
    for i, (model, acc, param, color, marker, size) in enumerate(zip(models, accuracies, params_m, colors, markers, sizes)):
        ax.scatter(param, acc, c=color, marker=marker, s=size, alpha=0.7, edgecolors='black', linewidth=1.5)
        ax.annotate(model, (param, acc), xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Parameters (M)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy vs Model Complexity')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#2ecc71', markersize=15, label='Teacher'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=10, label='Baseline'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#e74c3c', markersize=10, label='Distilled')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved accuracy vs complexity plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_pareto_frontier(
    results: Dict,
    save_path: Optional[str] = None,
    show: bool = True,
    highlight_optimal: bool = True
):
    """
    Plot Pareto frontier for model efficiency analysis.
    
    Generates multiple plots showing trade-offs between:
    - Accuracy vs Model Size
    - Accuracy vs Inference Time
    - Model Size vs Inference Time (color-coded by accuracy)
    
    Args:
        results: Dictionary with model results from benchmark_all_models.py
                 Each entry should have 'accuracy', 'size', and 'performance' keys
        save_path: Path to save figure (optional)
        show: Whether to display the plot
        highlight_optimal: Whether to highlight Pareto-optimal points
    
    Example results format:
        {
            'MobileNetV2 FP32': {
                'accuracy': {'accuracy': 99.5},
                'size': {'disk_size_mb': 8.5},
                'performance': {'per_sample_latency_ms': {'mean': 2.1}}
            },
            ...
        }
    """
    # Extract data
    models = []
    accuracies = []
    sizes = []
    latencies = []
    model_types = []  # For color coding
    
    for model_name, result in results.items():
        models.append(model_name)
        accuracies.append(result['accuracy']['accuracy'])
        sizes.append(result['size']['disk_size_mb'])
        latencies.append(result['performance']['per_sample_latency_ms']['mean'])
        
        # Determine model type for color coding
        if 'teacher' in model_name.lower() or 'se_resnet' in model_name.lower():
            model_types.append('Teacher')
        elif 'int8' in result.get('type', '').lower() or 'qat' in model_name.lower() or 'ptq' in model_name.lower():
            model_types.append('Quantized')
        elif 'distilled' in model_name.lower():
            model_types.append('Distilled')
        elif 'baseline' in model_name.lower():
            model_types.append('Baseline')
        else:
            model_types.append('Other')
    
    # Convert to numpy arrays
    accuracies = np.array(accuracies)
    sizes = np.array(sizes)
    latencies = np.array(latencies)
    
    # Define color map
    type_colors = {
        'Teacher': '#e74c3c',      # Red
        'Baseline': '#95a5a6',     # Gray
        'Distilled': '#3498db',    # Blue
        'Quantized': '#2ecc71',    # Green
        'Other': '#f39c12'         # Orange
    }
    colors = [type_colors[t] for t in model_types]
    
    # Compute Pareto frontier for accuracy vs size
    pareto_mask_size = np.ones(len(models), dtype=bool)
    if highlight_optimal:
        for i in range(len(models)):
            for j in range(len(models)):
                if i != j:
                    # Model j dominates model i if it has higher accuracy AND smaller size
                    if accuracies[j] >= accuracies[i] and sizes[j] <= sizes[i]:
                        if accuracies[j] > accuracies[i] or sizes[j] < sizes[i]:
                            pareto_mask_size[i] = False
                            break
    
    # Compute Pareto frontier for accuracy vs latency
    pareto_mask_latency = np.ones(len(models), dtype=bool)
    if highlight_optimal:
        for i in range(len(models)):
            for j in range(len(models)):
                if i != j:
                    # Model j dominates model i if it has higher accuracy AND lower latency
                    if accuracies[j] >= accuracies[i] and latencies[j] <= latencies[i]:
                        if accuracies[j] > accuracies[i] or latencies[j] < latencies[i]:
                            pareto_mask_latency[i] = False
                            break
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: Accuracy vs Model Size
    ax1 = plt.subplot(1, 3, 1)
    for i, (model, acc, size, color, model_type) in enumerate(zip(models, accuracies, sizes, colors, model_types)):
        marker = 'o'
        markersize = 10
        alpha = 0.7
        
        # Highlight Pareto-optimal points
        if highlight_optimal and pareto_mask_size[i]:
            marker = '*'
            markersize = 15
            alpha = 1.0
        
        ax1.scatter(size, acc, c=color, marker=marker, s=markersize**2, 
                   alpha=alpha, edgecolors='black', linewidth=1.5,
                   label=model_type if model_type not in [m[1] for m in zip(models[:i], model_types[:i])] else "")
    
    ax1.set_xlabel('Model Size (MB)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy vs Model Size\n(★ = Pareto Optimal)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Add model labels
    for i, (model, acc, size) in enumerate(zip(models, accuracies, sizes)):
        if highlight_optimal and pareto_mask_size[i]:
            # Only label Pareto-optimal points for clarity
            ax1.annotate(model.split()[0], (size, acc), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
    
    # Plot 2: Accuracy vs Inference Time
    ax2 = plt.subplot(1, 3, 2)
    for i, (model, acc, latency, color, model_type) in enumerate(zip(models, accuracies, latencies, colors, model_types)):
        marker = 'o'
        markersize = 10
        alpha = 0.7
        
        # Highlight Pareto-optimal points
        if highlight_optimal and pareto_mask_latency[i]:
            marker = '*'
            markersize = 15
            alpha = 1.0
        
        ax2.scatter(latency, acc, c=color, marker=marker, s=markersize**2,
                   alpha=alpha, edgecolors='black', linewidth=1.5)
    
    ax2.set_xlabel('Inference Time (ms/sample)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy vs Inference Speed\n(★ = Pareto Optimal)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # Add model labels
    for i, (model, acc, latency) in enumerate(zip(models, accuracies, latencies)):
        if highlight_optimal and pareto_mask_latency[i]:
            ax2.annotate(model.split()[0], (latency, acc),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
    
    # Plot 3: Size vs Latency (color = accuracy)
    ax3 = plt.subplot(1, 3, 3)
    scatter = ax3.scatter(sizes, latencies, c=accuracies, cmap='RdYlGn',
                         s=200, alpha=0.7, edgecolors='black', linewidth=1.5,
                         vmin=accuracies.min() - 1, vmax=accuracies.max())
    
    # Highlight Pareto-optimal in both dimensions
    pareto_both = pareto_mask_size & pareto_mask_latency
    if highlight_optimal and pareto_both.any():
        ax3.scatter(sizes[pareto_both], latencies[pareto_both],
                   marker='*', s=400, facecolors='none', edgecolors='red',
                   linewidth=3, label='Pareto Optimal')
    
    ax3.set_xlabel('Model Size (MB)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Inference Time (ms/sample)', fontsize=12, fontweight='bold')
    ax3.set_title('Efficiency Landscape\n(Color = Accuracy)', fontsize=14, fontweight='bold')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Accuracy (%)', fontsize=10, fontweight='bold')
    
    # Add model labels
    for i, (model, size, latency) in enumerate(zip(models, sizes, latencies)):
        if highlight_optimal and pareto_both[i]:
            ax3.annotate(model.split()[0], (size, latency),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8, fontweight='bold')
    
    # Add legend to first plot
    handles, labels = ax1.get_legend_handles_labels()
    # Remove duplicates
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Pareto frontier plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # Print Pareto-optimal models
    if highlight_optimal:
        print("\n" + "=" * 80)
        print("PARETO-OPTIMAL MODELS")
        print("=" * 80)
        print("\nAccuracy vs Size:")
        for i, model in enumerate(models):
            if pareto_mask_size[i]:
                print(f"  ★ {model}: {accuracies[i]:.2f}% accuracy, {sizes[i]:.2f} MB")
        
        print("\nAccuracy vs Latency:")
        for i, model in enumerate(models):
            if pareto_mask_latency[i]:
                print(f"  ★ {model}: {accuracies[i]:.2f}% accuracy, {latencies[i]:.2f} ms")
        
        print("\nBoth Dimensions:")
        for i, model in enumerate(models):
            if pareto_both[i]:
                print(f"  ★★ {model}: {accuracies[i]:.2f}% accuracy, {sizes[i]:.2f} MB, {latencies[i]:.2f} ms")
        print("=" * 80)


def create_visualization_report(experiment_dir: str, output_dir: Optional[str] = None):
    """
    Create comprehensive visualization report for an experiment
    
    Args:
        experiment_dir: Directory containing experiment results
        output_dir: Output directory for visualizations (default: experiment_dir/visualizations)
    """
    experiment_path = Path(experiment_dir)
    
    if output_dir is None:
        output_dir = experiment_path / 'visualizations'
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Creating Visualization Report for {experiment_path.name}")
    print(f"{'='*80}\n")
    
    # Plot training curves
    history_file = experiment_path / 'logs' / 'training_history.json'
    if history_file.exists():
        plot_training_curves(str(history_file), 
                           save_path=str(output_path / 'training_curves.png'),
                           show=False)
    
    # Plot KD loss components if available
    if history_file.exists():
        plot_kd_loss_components(str(history_file),
                               save_path=str(output_path / 'kd_loss_components.png'),
                               show=False)
    
    print(f"\n✓ Visualization report created in: {output_path}")


# Test the visualization tools
if __name__ == '__main__':
    print("=" * 80)
    print("VISUALIZATION TOOLS TEST")
    print("=" * 80)
    
    # Create sample data for testing
    print("\nCreating sample visualizations...")
    
    # Sample confusion matrix
    cm = np.array([[45, 2, 1, 2],
                   [3, 48, 1, 0],
                   [1, 2, 46, 1],
                   [2, 0, 1, 47]])
    class_names = ['Normal', 'Inner Race', 'Outer Race', 'Ball']
    
    plot_confusion_matrix(cm, class_names, show=False)
    print("✓ Confusion matrix plot created")
    
    # Sample model comparison
    sample_results = {
        'Teacher (SE-ResNet152)': {
            'accuracy': 99.75,
            'model_info': {'total_parameters_M': 64.78},
            'inference_time_ms': 45.2
        },
        'MobileNetV2 (Baseline)': {
            'accuracy': 97.80,
            'model_info': {'total_parameters_M': 2.23},
            'inference_time_ms': 2.1
        },
        'MobileNetV2 (Distilled)': {
            'accuracy': 99.50,
            'model_info': {'total_parameters_M': 2.23},
            'inference_time_ms': 2.1
        }
    }
    
    plot_model_comparison(sample_results, show=False)
    print("✓ Model comparison plot created")
    
    plot_accuracy_vs_complexity(sample_results, show=False)
    print("✓ Accuracy vs complexity plot created")
    
    print("\n✓ Visualization tools working correctly!")

