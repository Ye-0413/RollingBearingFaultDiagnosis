"""
Model Quantization Script

This script quantizes trained FP32 models to INT8 precision using
Post-Training Quantization (PTQ) methods.

Supports:
- Dynamic Quantization: Fast, weights-only quantization
- Static Quantization: Full quantization with calibration data

Usage:
    # Dynamic quantization (quick, no calibration needed)
    python quantize_model.py \
        --model mobilenetv2 \
        --checkpoint experiments/distilled/mobilenetv2_load0/checkpoints/best_model.pth \
        --method dynamic \
        --output quantized_models/mobilenetv2_dynamic.pth
    
    # Static quantization (better accuracy, requires calibration)
    python quantize_model.py \
        --model mobilenetv2 \
        --checkpoint experiments/distilled/mobilenetv2_load0/checkpoints/best_model.pth \
        --method static \
        --calibration-data Training_data/load_0/train.txt \
        --calibration-samples 1000 \
        --output quantized_models/mobilenetv2_static.pth

Author: Research Team
Date: 2025-11-13
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.quantization import PostTrainingQuantizer, get_calibration_loader
from src.models.mobilenetv2 import mobilenet_v2
from src.models.resnet18_light import resnet18


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Quantize trained models to INT8 precision',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dynamic quantization (fastest)
  python quantize_model.py --model mobilenetv2 --checkpoint path/to/model.pth --method dynamic

  # Static quantization (best accuracy)
  python quantize_model.py --model mobilenetv2 --checkpoint path/to/model.pth --method static \\
      --calibration-data Training_data/load_0/train.txt --calibration-samples 1000
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['mobilenetv2', 'resnet18'],
        help='Model architecture to quantize'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained FP32 model checkpoint'
    )
    parser.add_argument(
        '--method',
        type=str,
        required=True,
        choices=['dynamic', 'static'],
        help='Quantization method (dynamic or static)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--num-classes',
        type=int,
        default=4,
        help='Number of output classes (default: 4)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for quantized model (auto-generated if not specified)'
    )
    parser.add_argument(
        '--backend',
        type=str,
        default='fbgemm',
        choices=['fbgemm', 'qnnpack'],
        help='Quantization backend (fbgemm for x86, qnnpack for ARM)'
    )
    
    # Static quantization specific
    parser.add_argument(
        '--calibration-data',
        type=str,
        default=None,
        help='Path to calibration data annotation file (required for static quantization)'
    )
    parser.add_argument(
        '--calibration-samples',
        type=int,
        default=1000,
        help='Number of samples for calibration (default: 1000)'
    )
    parser.add_argument(
        '--calibration-batch-size',
        type=int,
        default=32,
        help='Batch size for calibration (default: 32)'
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to run on (cpu or cuda)'
    )
    
    args = parser.parse_args()
    
    # Validate static quantization arguments
    if args.method == 'static' and args.calibration_data is None:
        parser.error("--calibration-data is required for static quantization")
    
    return args


def load_model(model_name: str, num_classes: int, checkpoint_path: str, device: str) -> nn.Module:
    """
    Load a trained FP32 model.
    
    Args:
        model_name: Model architecture name
        num_classes: Number of output classes
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    print(f"Loading {model_name} model...")
    
    # Create model architecture
    if model_name == 'mobilenetv2':
        model = mobilenet_v2(num_classes=num_classes)
    elif model_name == 'resnet18':
        model = resnet18(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from {checkpoint_path}")
    return model


def quantize_model(
    model: nn.Module,
    method: str,
    backend: str,
    device: str,
    calibration_loader=None
) -> nn.Module:
    """
    Quantize a model using the specified method.
    
    Args:
        model: FP32 model to quantize
        method: 'dynamic' or 'static'
        backend: Quantization backend
        device: Device to run on
        calibration_loader: DataLoader for calibration (required for static)
    
    Returns:
        Quantized model
    """
    print(f"\nQuantizing model using {method} quantization...")
    print(f"Backend: {backend}")
    
    # Create quantizer
    quantizer = PostTrainingQuantizer(model, backend=backend, device=device)
    
    # Apply quantization
    if method == 'dynamic':
        quantized_model = quantizer.quantize_dynamic()
    elif method == 'static':
        if calibration_loader is None:
            raise ValueError("Calibration loader required for static quantization")
        quantized_model = quantizer.quantize_static(calibration_loader)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print("✓ Quantization completed")
    
    # Compare model sizes
    print("\nModel size comparison:")
    size_comparison = quantizer.compare_model_sizes(model, quantized_model)
    print(f"  FP32:        {size_comparison['fp32_size_mb']:.2f} MB")
    print(f"  INT8:        {size_comparison['quantized_size_mb']:.2f} MB")
    print(f"  Compression: {size_comparison['compression_ratio']:.2f}x")
    print(f"  Reduction:   {size_comparison['size_reduction_percent']:.1f}%")
    
    return quantized_model


def save_quantized_model(
    quantized_model: nn.Module,
    output_path: str,
    metadata: dict
):
    """
    Save quantized model with metadata.
    
    Args:
        quantized_model: Quantized model to save
        output_path: Output file path
        metadata: Metadata dictionary
    """
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save model
    torch.save(quantized_model.state_dict(), output_path)
    
    # Save metadata
    metadata_path = output_path.replace('.pth', '_metadata.pth')
    torch.save(metadata, metadata_path)
    
    print(f"\n✓ Quantized model saved to: {output_path}")
    print(f"✓ Metadata saved to: {metadata_path}")


def main():
    """Main execution function."""
    args = parse_args()
    
    print("=" * 70)
    print("MODEL QUANTIZATION SCRIPT")
    print("=" * 70)
    print(f"Model:      {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Method:     {args.method}")
    print(f"Backend:    {args.backend}")
    print(f"Device:     {args.device}")
    print("=" * 70)
    
    # Step 1: Load FP32 model
    model = load_model(
        args.model,
        args.num_classes,
        args.checkpoint,
        args.device
    )
    
    # Step 2: Prepare calibration data (if static quantization)
    calibration_loader = None
    if args.method == 'static':
        print(f"\nPreparing calibration data...")
        print(f"  Source: {args.calibration_data}")
        print(f"  Samples: {args.calibration_samples}")
        print(f"  Batch size: {args.calibration_batch_size}")
        
        calibration_loader = get_calibration_loader(
            annotation_file=args.calibration_data,
            num_samples=args.calibration_samples,
            batch_size=args.calibration_batch_size,
            shuffle=True
        )
        print(f"✓ Calibration data ready")
    
    # Step 3: Quantize model
    quantized_model = quantize_model(
        model,
        args.method,
        args.backend,
        args.device,
        calibration_loader
    )
    
    # Step 4: Generate output path if not specified
    if args.output is None:
        checkpoint_name = Path(args.checkpoint).stem
        output_dir = "quantized_models"
        args.output = f"{output_dir}/{args.model}_{checkpoint_name}_{args.method}.pth"
    
    # Step 5: Save quantized model
    metadata = {
        'model': args.model,
        'num_classes': args.num_classes,
        'method': args.method,
        'backend': args.backend,
        'original_checkpoint': args.checkpoint,
        'calibration_samples': args.calibration_samples if args.method == 'static' else None
    }
    
    save_quantized_model(quantized_model, args.output, metadata)
    
    print("\n" + "=" * 70)
    print("QUANTIZATION COMPLETE!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Evaluate accuracy: python src/evaluate_quantized.py --model {args.output}")
    print(f"  2. Benchmark performance: python benchmark_all_models.py")
    print(f"  3. Deploy to edge device: python deploy_to_edge.py --model {args.output}")
    print()


if __name__ == "__main__":
    main()

