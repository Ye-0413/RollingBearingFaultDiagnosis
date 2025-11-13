"""
Quantization Module for Model Compression

This module provides tools for quantizing trained models to INT8 precision,
reducing model size by ~4x while maintaining accuracy.

Available quantization methods:
- Dynamic Quantization: Quantize weights only (fast, no calibration needed)
- Static Quantization: Quantize weights and activations (requires calibration)
- Quantization-Aware Training (QAT): Train with quantization simulation

Classes:
    PostTrainingQuantizer: Main class for PTQ (dynamic and static)
    
Usage:
    from src.quantization import PostTrainingQuantizer
    
    quantizer = PostTrainingQuantizer(model, backend='fbgemm')
    quantized_model = quantizer.quantize_static(calibration_loader)
"""

from .post_training_quantization import PostTrainingQuantizer

__all__ = ['PostTrainingQuantizer']

