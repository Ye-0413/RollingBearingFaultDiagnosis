"""
Post-Training Quantization (PTQ) for PyTorch Models

This module implements dynamic and static quantization methods for
compressing trained FP32 models to INT8 precision.

Key Features:
- Dynamic Quantization: Fast, weights-only quantization
- Static Quantization: Full quantization with calibration
- Automatic backend selection (fbgemm for x86, qnnpack for ARM)
- Model preparation and conversion utilities

Author: Research Team
Date: 2025-11-13
"""

import os
import copy
import torch
import torch.nn as nn
import torch.quantization as quant
from torch.utils.data import DataLoader
from typing import Optional, Callable, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PostTrainingQuantizer:
    """
    Post-Training Quantization wrapper for PyTorch models.
    
    Supports both dynamic and static quantization methods.
    
    Args:
        model (nn.Module): The FP32 model to quantize
        backend (str): Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
        device (str): Device to run calibration on ('cpu' or 'cuda')
    
    Example:
        >>> model = mobilenet_v2(num_classes=4)
        >>> model.load_state_dict(torch.load('checkpoint.pth'))
        >>> quantizer = PostTrainingQuantizer(model, backend='fbgemm')
        >>> quantized_model = quantizer.quantize_static(calibration_loader)
    """
    
    def __init__(
        self,
        model: nn.Module,
        backend: str = 'fbgemm',
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.backend = backend
        self.device = device
        
        # Set quantization backend
        torch.backends.quantized.engine = backend
        logger.info(f"Initialized PostTrainingQuantizer with backend: {backend}")
    
    def quantize_dynamic(
        self,
        qconfig_spec: Optional[Dict] = None
    ) -> nn.Module:
        """
        Dynamic quantization: Quantize weights only, activations stay FP32.
        
        This is the fastest quantization method and works well for models
        with many Linear/LSTM layers. No calibration data needed.
        
        Args:
            qconfig_spec: Optional dict specifying which layers to quantize.
                         If None, quantizes all Linear and Conv2d layers.
        
        Returns:
            Quantized model (nn.Module)
        
        Example:
            >>> quantized_model = quantizer.quantize_dynamic()
            >>> # Model size reduced by ~4x, inference speed improved
        """
        logger.info("Starting dynamic quantization...")
        
        # Create a copy to avoid modifying the original
        model_to_quantize = copy.deepcopy(self.model)
        model_to_quantize.eval()
        
        # Default: quantize Linear and Conv2d layers
        if qconfig_spec is None:
            qconfig_spec = {nn.Linear, nn.Conv2d}
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model_to_quantize,
            qconfig_spec,
            dtype=torch.qint8
        )
        
        logger.info("Dynamic quantization completed successfully")
        return quantized_model
    
    def quantize_static(
        self,
        calibration_loader: DataLoader,
        num_calibration_batches: Optional[int] = None
    ) -> nn.Module:
        """
        Static quantization: Quantize both weights and activations.
        
        This provides better compression and speedup than dynamic quantization,
        but requires calibration data to compute activation ranges.
        
        Args:
            calibration_loader: DataLoader providing calibration samples
            num_calibration_batches: Number of batches to use for calibration
                                    If None, uses all data in loader
        
        Returns:
            Quantized model (nn.Module)
        
        Example:
            >>> calibration_loader = DataLoader(calib_dataset, batch_size=32)
            >>> quantized_model = quantizer.quantize_static(calibration_loader)
        """
        logger.info("Starting static quantization...")
        
        # Step 1: Prepare model for quantization
        model_to_quantize = copy.deepcopy(self.model)
        model_to_quantize.eval()
        
        # Fuse modules for better performance (Conv+BN+ReLU → single op)
        model_to_quantize = self._fuse_modules(model_to_quantize)
        
        # Set quantization configuration
        model_to_quantize.qconfig = quant.get_default_qconfig(self.backend)
        
        # Prepare model (insert observers)
        quant.prepare(model_to_quantize, inplace=True)
        logger.info("Model prepared, running calibration...")
        
        # Step 2: Calibration - run inference to collect statistics
        model_to_quantize.to(self.device)
        self._calibrate(model_to_quantize, calibration_loader, num_calibration_batches)
        
        # Step 3: Convert to quantized model
        logger.info("Converting to quantized model...")
        quantized_model = quant.convert(model_to_quantize, inplace=True)
        
        logger.info("Static quantization completed successfully")
        return quantized_model
    
    def _fuse_modules(self, model: nn.Module) -> nn.Module:
        """
        Fuse adjacent modules for better quantization performance.
        
        Common fusions:
        - Conv2d + BatchNorm2d + ReLU
        - Conv2d + BatchNorm2d
        - Linear + ReLU
        
        This is model-specific and may need customization.
        """
        logger.info("Fusing modules...")
        
        # Try to fuse common patterns
        # Note: This is a generic implementation. For specific models,
        # you may need to specify exact module names to fuse.
        
        try:
            # For MobileNetV2 and similar architectures
            # This will fuse Conv-BN-ReLU sequences
            for module_name, module in model.named_children():
                if hasattr(module, 'fuse_model'):
                    module.fuse_model()
        except Exception as e:
            logger.warning(f"Could not fuse modules automatically: {e}")
            logger.info("Continuing without fusion (may result in slightly lower performance)")
        
        return model
    
    def _calibrate(
        self,
        model: nn.Module,
        calibration_loader: DataLoader,
        num_batches: Optional[int] = None
    ):
        """
        Run calibration to collect activation statistics.
        
        Args:
            model: Model with observers inserted
            calibration_loader: DataLoader with calibration data
            num_batches: Maximum number of batches to use
        """
        model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(calibration_loader):
                if num_batches is not None and batch_idx >= num_batches:
                    break
                
                # Handle different data formats
                if isinstance(batch, (tuple, list)):
                    images = batch[0]
                else:
                    images = batch
                
                images = images.to(self.device)
                
                # Forward pass to collect statistics
                _ = model(images)
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Calibrated {batch_idx + 1} batches")
        
        logger.info(f"Calibration completed with {batch_idx + 1} batches")
    
    def save_quantized_model(
        self,
        quantized_model: nn.Module,
        save_path: str
    ):
        """
        Save quantized model to disk.
        
        Args:
            quantized_model: The quantized model to save
            save_path: Path to save the model
        
        Example:
            >>> quantizer.save_quantized_model(
            ...     quantized_model,
            ...     'quantized_models/mobilenetv2_int8.pth'
            ... )
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the entire model (not just state_dict for quantized models)
        torch.save(quantized_model.state_dict(), save_path)
        
        # Also save metadata
        metadata = {
            'backend': self.backend,
            'quantization_method': 'static',
            'dtype': 'int8'
        }
        metadata_path = save_path.replace('.pth', '_metadata.pth')
        torch.save(metadata, metadata_path)
        
        logger.info(f"Quantized model saved to {save_path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    @staticmethod
    def load_quantized_model(
        model_architecture: nn.Module,
        checkpoint_path: str,
        device: str = 'cpu'
    ) -> nn.Module:
        """
        Load a quantized model from disk.
        
        Args:
            model_architecture: An instance of the model architecture
            checkpoint_path: Path to the saved quantized model
            device: Device to load the model on
        
        Returns:
            Loaded quantized model
        
        Example:
            >>> from src.models.mobilenetv2 import mobilenet_v2
            >>> model_arch = mobilenet_v2(num_classes=4)
            >>> quantized_model = PostTrainingQuantizer.load_quantized_model(
            ...     model_arch,
            ...     'quantized_models/mobilenetv2_int8.pth'
            ... )
        """
        model_architecture.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model_architecture.to(device)
        model_architecture.eval()
        
        logger.info(f"Loaded quantized model from {checkpoint_path}")
        return model_architecture
    
    def compare_model_sizes(
        self,
        fp32_model: nn.Module,
        quantized_model: nn.Module
    ) -> Dict[str, Any]:
        """
        Compare the sizes of FP32 and quantized models.
        
        Args:
            fp32_model: Original FP32 model
            quantized_model: Quantized INT8 model
        
        Returns:
            Dictionary with size comparison statistics
        """
        import tempfile
        
        # Save models temporarily to measure size
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=True) as f:
            torch.save(fp32_model.state_dict(), f.name)
            fp32_size = os.path.getsize(f.name)
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=True) as f:
            torch.save(quantized_model.state_dict(), f.name)
            quantized_size = os.path.getsize(f.name)
        
        compression_ratio = fp32_size / quantized_size
        
        results = {
            'fp32_size_mb': fp32_size / (1024 * 1024),
            'quantized_size_mb': quantized_size / (1024 * 1024),
            'compression_ratio': compression_ratio,
            'size_reduction_percent': (1 - 1/compression_ratio) * 100
        }
        
        logger.info(f"Model size comparison:")
        logger.info(f"  FP32: {results['fp32_size_mb']:.2f} MB")
        logger.info(f"  INT8: {results['quantized_size_mb']:.2f} MB")
        logger.info(f"  Compression: {compression_ratio:.2f}x")
        
        return results


def get_calibration_loader(
    annotation_file: str,
    num_samples: int = 1000,
    batch_size: int = 32,
    shuffle: bool = True
) -> DataLoader:
    """
    Create a DataLoader for calibration data.
    
    Args:
        annotation_file: Path to annotation file (e.g., train.txt)
        num_samples: Maximum number of samples to use for calibration
        batch_size: Batch size for calibration
        shuffle: Whether to shuffle the data
    
    Returns:
        DataLoader for calibration
    
    Example:
        >>> calib_loader = get_calibration_loader(
        ...     'Training_data/load_0/train.txt',
        ...     num_samples=1000,
        ...     batch_size=32
        ... )
    """
    from src.data.dataset_loader import BearingDataset
    from torch.utils.data import Subset
    import numpy as np
    
    # Load full dataset
    full_dataset = BearingDataset(annotation_file=annotation_file)
    
    # Subsample if needed
    if num_samples < len(full_dataset):
        indices = np.random.choice(len(full_dataset), num_samples, replace=False)
        calibration_dataset = Subset(full_dataset, indices)
    else:
        calibration_dataset = full_dataset
    
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Created calibration loader with {len(calibration_dataset)} samples")
    return calibration_loader


if __name__ == "__main__":
    # Quick test
    print("Post-Training Quantization Module")
    print("=" * 60)
    print("Available quantization methods:")
    print("  1. Dynamic Quantization (weights only)")
    print("  2. Static Quantization (weights + activations)")
    print()
    print("Usage:")
    print("  from src.quantization import PostTrainingQuantizer")
    print("  quantizer = PostTrainingQuantizer(model)")
    print("  quantized_model = quantizer.quantize_static(calib_loader)")

