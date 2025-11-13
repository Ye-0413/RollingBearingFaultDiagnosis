"""
Model Factory for Knowledge Distillation Framework

Unified interface for creating teacher and student models.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn

# Import SEResNet directly from seresnet.py
from configs.backbones.seresnet import SEResNet

# Import student models
from src.models.mobilenetv2 import mobilenet_v2
from src.models.resnet18_light import resnet18


class TeacherModelWrapper(nn.Module):
    """
    Wrapper for SE-ResNet152 teacher model to provide unified interface.
    
    The original SEResNet returns feature maps, this wrapper adds a classifier
    and provides both logits and intermediate features for distillation.
    """
    
    def __init__(self, num_classes=4, pretrained_path=None):
        super(TeacherModelWrapper, self).__init__()
        
        # Create SE-ResNet152 backbone
        self.backbone = SEResNet(
            depth=152,
            num_stages=4,
            out_indices=(0, 1, 2, 3),  # Get all stage outputs
            style='pytorch',
            se_ratio=16
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier head
        self.fc = nn.Linear(2048, num_classes)
        
        # Load pretrained weights if provided
        if pretrained_path and os.path.exists(pretrained_path):
            self.load_pretrained(pretrained_path)
            
        self.intermediate_features = {}
    
    def load_pretrained(self, checkpoint_path):
        """Load pretrained weights from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Load weights (may need to adjust key names)
        try:
            self.load_state_dict(state_dict, strict=False)
            print(f"✓ Loaded pretrained weights from {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    def forward(self, x, return_features=False):
        """
        Forward pass
        
        Args:
            x: Input tensor [B, 3, H, W]
            return_features: If True, return intermediate features for distillation
        
        Returns:
            logits: Classification logits [B, num_classes]
            features (optional): Dict of intermediate features
        """
        # Get backbone features
        backbone_outputs = self.backbone(x)
        
        if return_features:
            # Store intermediate features
            self.intermediate_features = {
                'stage1': backbone_outputs[0],
                'stage2': backbone_outputs[1],
                'stage3': backbone_outputs[2],
                'stage4': backbone_outputs[3]
            }
        
        # Use last stage for classification
        x = backbone_outputs[-1]
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        if return_features:
            self.intermediate_features['final'] = x
        
        logits = self.fc(x)
        
        if return_features:
            return logits, self.intermediate_features
        
        return logits
    
    def get_parameter_count(self):
        """Return number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameter_count(self):
        """Return number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(model_name, num_classes=4, pretrained=False, pretrained_path=None):
    """
    Factory function to create models
    
    Args:
        model_name: Name of the model ('se_resnet152', 'mobilenetv2', 'resnet18')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        pretrained_path: Path to pretrained checkpoint
    
    Returns:
        model: PyTorch model
    """
    model_name = model_name.lower()
    
    if model_name == 'se_resnet152' or model_name == 'seresnet152':
        model = TeacherModelWrapper(
            num_classes=num_classes,
            pretrained_path=pretrained_path if pretrained else None
        )
        print(f"✓ Created SE-ResNet152 teacher model")
        
    elif model_name == 'mobilenetv2' or model_name == 'mobilenet_v2':
        model = mobilenet_v2(
            num_classes=num_classes,
            pretrained=pretrained
        )
        print(f"✓ Created MobileNetV2 student model")
        
    elif model_name == 'resnet18':
        model = resnet18(
            num_classes=num_classes,
            pretrained=pretrained
        )
        print(f"✓ Created ResNet18 student model")
        
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Available models: se_resnet152, mobilenetv2, resnet18")
    
    return model


def get_model_info(model):
    """
    Get detailed information about a model
    
    Args:
        model: PyTorch model
    
    Returns:
        info: Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'total_parameters_M': total_params / 1e6,
        'trainable_parameters_M': trainable_params / 1e6,
        'model_size_mb': total_params * 4 / (1024 ** 2),  # Assuming float32
    }
    
    return info


if __name__ == '__main__':
    print("=" * 80)
    print("MODEL FACTORY TEST")
    print("=" * 80)
    
    # Test all models
    models_to_test = ['se_resnet152', 'mobilenetv2', 'resnet18']
    
    for model_name in models_to_test:
        print(f"\n{'=' * 80}")
        print(f"Testing {model_name.upper()}")
        print('=' * 80)
        
        model = create_model(model_name, num_classes=4)
        info = get_model_info(model)
        
        print(f"\nModel Info:")
        print(f"  Total Parameters: {info['total_parameters']:,}")
        print(f"  Trainable Parameters: {info['trainable_parameters']:,}")
        print(f"  Parameters (M): {info['total_parameters_M']:.2f}M")
        print(f"  Model Size: {info['model_size_mb']:.2f} MB")
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        
        # Test without features
        output = model(x)
        print(f"\nForward Pass (no features):")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Test with features
        output, features = model(x, return_features=True)
        print(f"\nForward Pass (with features):")
        print(f"  Output shape: {output.shape}")
        print(f"  Number of feature maps: {len(features)}")
        for key, feat in features.items():
            if isinstance(feat, torch.Tensor):
                print(f"    {key}: {feat.shape}")
    
    print(f"\n{'=' * 80}")
    print("✓ ALL MODELS TESTED SUCCESSFULLY")
    print('=' * 80)

