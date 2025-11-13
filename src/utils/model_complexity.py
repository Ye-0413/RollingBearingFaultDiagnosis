"""
Model Complexity Analysis Utilities

Provides tools for analyzing model complexity:
- Parameter count
- FLOPs (Floating Point Operations)
- Model size
- Inference time
- Memory footprint
"""

import time
import torch
import torch.nn as nn
from typing import Tuple, Dict, List
import numpy as np


class FLOPsCounter:
    """
    Count FLOPs for PyTorch models
    
    Supports common layers: Conv2d, Linear, BatchNorm2d, etc.
    """
    
    def __init__(self):
        self.total_flops = 0
        self.layer_flops = {}
        self.hooks = []
    
    def count_conv2d(self, module, input, output):
        """Count FLOPs for Conv2d layer"""
        batch_size, in_channels, in_h, in_w = input[0].shape
        out_channels, out_h, out_w = output.shape[1:]
        
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (in_channels // module.groups)
        output_size = out_channels * out_h * out_w
        
        # Multiply-accumulate operations
        flops = batch_size * kernel_ops * output_size
        
        # Add bias if present
        if module.bias is not None:
            flops += batch_size * out_channels * out_h * out_w
        
        self.total_flops += flops
        self.layer_flops[module] = flops
    
    def count_linear(self, module, input, output):
        """Count FLOPs for Linear layer"""
        batch_size = input[0].shape[0]
        in_features = module.in_features
        out_features = module.out_features
        
        # Multiply-accumulate operations
        flops = batch_size * in_features * out_features
        
        # Add bias if present
        if module.bias is not None:
            flops += batch_size * out_features
        
        self.total_flops += flops
        self.layer_flops[module] = flops
    
    def count_batchnorm2d(self, module, input, output):
        """Count FLOPs for BatchNorm2d layer"""
        batch_size, channels, height, width = input[0].shape
        
        # Normalization: subtract mean, divide by std (2 ops per element)
        # Scale and shift: multiply by gamma, add beta (2 ops per element)
        flops = batch_size * channels * height * width * 4
        
        self.total_flops += flops
        self.layer_flops[module] = flops
    
    def count_relu(self, module, input, output):
        """Count FLOPs for ReLU activation"""
        # ReLU is essentially free (comparison operation)
        flops = output.numel()
        self.total_flops += flops
        self.layer_flops[module] = flops
    
    def count_pooling(self, module, input, output):
        """Count FLOPs for pooling layers"""
        # Pooling: kernel_size comparisons per output element
        if hasattr(module, 'kernel_size'):
            if isinstance(module.kernel_size, tuple):
                kernel_size = module.kernel_size[0] * module.kernel_size[1]
            else:
                kernel_size = module.kernel_size * module.kernel_size
        else:
            kernel_size = 1
        
        flops = output.numel() * kernel_size
        self.total_flops += flops
        self.layer_flops[module] = flops
    
    def register_hooks(self, model):
        """Register forward hooks to count FLOPs"""
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                hook = module.register_forward_hook(self.count_conv2d)
                self.hooks.append(hook)
            elif isinstance(module, nn.Linear):
                hook = module.register_forward_hook(self.count_linear)
                self.hooks.append(hook)
            elif isinstance(module, nn.BatchNorm2d):
                hook = module.register_forward_hook(self.count_batchnorm2d)
                self.hooks.append(hook)
            elif isinstance(module, (nn.ReLU, nn.ReLU6)):
                hook = module.register_forward_hook(self.count_relu)
                self.hooks.append(hook)
            elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                hook = module.register_forward_hook(self.count_pooling)
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def reset(self):
        """Reset FLOPs counter"""
        self.total_flops = 0
        self.layer_flops = {}


def count_flops(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 224, 224), 
                device: str = 'cpu') -> int:
    """
    Count total FLOPs for a model
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, channels, height, width)
        device: Device to run on
    
    Returns:
        total_flops: Total FLOPs count
    """
    model = model.to(device)
    model.eval()
    
    counter = FLOPsCounter()
    counter.register_hooks(model)
    
    # Create dummy input
    dummy_input = torch.randn(*input_size).to(device)
    
    # Forward pass to count FLOPs
    with torch.no_grad():
        _ = model(dummy_input)
    
    total_flops = counter.total_flops
    counter.remove_hooks()
    
    return total_flops


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count model parameters
    
    Args:
        model: PyTorch model
    
    Returns:
        total_params: Total parameters
        trainable_params: Trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def measure_inference_time(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
                          device: str = 'cpu', num_runs: int = 100, warmup_runs: int = 10) -> Dict[str, float]:
    """
    Measure model inference time
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
        device: Device to run on
        num_runs: Number of inference runs
        warmup_runs: Number of warmup runs
    
    Returns:
        timing_stats: Dictionary with timing statistics
    """
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(*input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
        'fps': float(1000.0 / np.mean(times))  # Frames per second
    }


def estimate_memory_footprint(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
                              device: str = 'cpu') -> Dict[str, float]:
    """
    Estimate model memory footprint
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
        device: Device to run on
    
    Returns:
        memory_stats: Dictionary with memory statistics (MB)
    """
    model = model.to(device)
    model.eval()
    
    # Parameter memory
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Buffer memory (e.g., BatchNorm running stats)
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
    
    # Estimate activation memory with a forward pass
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        dummy_input = torch.randn(*input_size).to(device)
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        peak_memory = torch.cuda.max_memory_allocated()
        activation_memory = peak_memory - param_memory - buffer_memory
    else:
        # Rough estimate for CPU (less accurate)
        activation_memory = 0
        dummy_input = torch.randn(*input_size)
        
        # Hook to track activation sizes
        activation_sizes = []
        
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                activation_sizes.append(output.numel() * output.element_size())
        
        hooks = []
        for module in model.modules():
            hooks.append(module.register_forward_hook(hook_fn))
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        for hook in hooks:
            hook.remove()
        
        activation_memory = sum(activation_sizes)
    
    return {
        'parameters_mb': param_memory / (1024 ** 2),
        'buffers_mb': buffer_memory / (1024 ** 2),
        'activations_mb': max(0, activation_memory) / (1024 ** 2),
        'total_mb': (param_memory + buffer_memory + max(0, activation_memory)) / (1024 ** 2)
    }


def analyze_model_complexity(model: nn.Module, model_name: str = "Model",
                             input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
                             device: str = 'cpu', verbose: bool = True) -> Dict:
    """
    Comprehensive model complexity analysis
    
    Args:
        model: PyTorch model
        model_name: Name for display
        input_size: Input tensor size
        device: Device to run on
        verbose: Print detailed results
    
    Returns:
        analysis: Dictionary with all complexity metrics
    """
    if verbose:
        print("=" * 80)
        print(f"MODEL COMPLEXITY ANALYSIS: {model_name}")
        print("=" * 80)
    
    # Parameter count
    total_params, trainable_params = count_parameters(model)
    
    if verbose:
        print(f"\n📊 PARAMETERS:")
        print(f"  Total:      {total_params:,} ({total_params / 1e6:.2f}M)")
        print(f"  Trainable:  {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
        print(f"  Frozen:     {total_params - trainable_params:,}")
    
    # FLOPs
    try:
        flops = count_flops(model, input_size, device)
        gflops = flops / 1e9
        
        if verbose:
            print(f"\n⚡ COMPUTATIONAL COMPLEXITY:")
            print(f"  FLOPs:      {flops:,} ({gflops:.2f} GFLOPs)")
            print(f"  Input:      {input_size[1]}x{input_size[2]}x{input_size[3]}")
    except Exception as e:
        if verbose:
            print(f"\n⚠ FLOPs counting failed: {e}")
        flops = 0
        gflops = 0
    
    # Inference time
    timing_stats = measure_inference_time(model, input_size, device)
    
    if verbose:
        print(f"\n⏱️  INFERENCE TIME ({device.upper()}):")
        print(f"  Mean:       {timing_stats['mean_ms']:.2f} ± {timing_stats['std_ms']:.2f} ms")
        print(f"  Median:     {timing_stats['median_ms']:.2f} ms")
        print(f"  Min/Max:    {timing_stats['min_ms']:.2f} / {timing_stats['max_ms']:.2f} ms")
        print(f"  P95/P99:    {timing_stats['p95_ms']:.2f} / {timing_stats['p99_ms']:.2f} ms")
        print(f"  FPS:        {timing_stats['fps']:.1f}")
    
    # Memory footprint
    memory_stats = estimate_memory_footprint(model, input_size, device)
    
    if verbose:
        print(f"\n💾 MEMORY FOOTPRINT:")
        print(f"  Parameters: {memory_stats['parameters_mb']:.2f} MB")
        print(f"  Buffers:    {memory_stats['buffers_mb']:.2f} MB")
        print(f"  Activations:{memory_stats['activations_mb']:.2f} MB (estimated)")
        print(f"  Total:      {memory_stats['total_mb']:.2f} MB")
    
    # Model size on disk (FP32)
    model_size_mb = (total_params * 4) / (1024 ** 2)
    
    if verbose:
        print(f"\n💿 MODEL SIZE (FP32):")
        print(f"  On Disk:    {model_size_mb:.2f} MB")
        print(f"  FP16:       {model_size_mb / 2:.2f} MB (estimated)")
        print(f"  INT8:       {model_size_mb / 4:.2f} MB (estimated)")
    
    # Efficiency metrics
    if flops > 0:
        params_per_gflop = total_params / gflops
        if verbose:
            print(f"\n📈 EFFICIENCY METRICS:")
            print(f"  Params/GFLOPs: {params_per_gflop / 1e6:.2f}M")
    
    if verbose:
        print("=" * 80)
    
    return {
        'model_name': model_name,
        'parameters': {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params,
            'total_M': total_params / 1e6
        },
        'flops': {
            'total': flops,
            'gflops': gflops
        },
        'inference_time': timing_stats,
        'memory': memory_stats,
        'model_size': {
            'fp32_mb': model_size_mb,
            'fp16_mb': model_size_mb / 2,
            'int8_mb': model_size_mb / 4
        },
        'input_size': input_size,
        'device': device
    }


# Test the utilities
if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.models.model_factory import create_model
    
    print("\n" + "=" * 80)
    print("MODEL COMPLEXITY ANALYSIS TOOL - TEST")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Test with all models
    models_to_test = [
        ('se_resnet152', 'SE-ResNet152 (Teacher)'),
        ('mobilenetv2', 'MobileNetV2 (Student)'),
        ('resnet18', 'ResNet18 (Student)')
    ]
    
    all_results = []
    
    for model_name, display_name in models_to_test:
        print("\n")
        model = create_model(model_name, num_classes=4)
        
        results = analyze_model_complexity(
            model,
            model_name=display_name,
            input_size=(1, 3, 224, 224),
            device=device,
            verbose=True
        )
        
        all_results.append(results)
    
    # Comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(f"\n{'Model':<30} {'Params (M)':<12} {'GFLOPs':<10} {'Time (ms)':<12} {'FPS':<8}")
    print("-" * 80)
    
    for result in all_results:
        print(f"{result['model_name']:<30} "
              f"{result['parameters']['total_M']:>11.2f} "
              f"{result['flops']['gflops']:>9.2f} "
              f"{result['inference_time']['mean_ms']:>11.2f} "
              f"{result['inference_time']['fps']:>7.1f}")
    
    print("\n" + "=" * 80)
    print("✓ MODEL COMPLEXITY ANALYSIS COMPLETE")
    print("=" * 80)

