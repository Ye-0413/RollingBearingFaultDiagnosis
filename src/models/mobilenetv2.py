"""
MobileNetV2 Implementation for Knowledge Distillation

Lightweight student model based on:
"MobileNetV2: Inverted Residuals and Linear Bottlenecks" (Sandler et al., 2018)

Parameters: ~3.5M (94.7% reduction vs SE-ResNet152)
"""

import torch
import torch.nn as nn
import math


def _make_divisible(v, divisor, min_value=None):
    """
    Ensure that all layers have a channel number divisible by 8
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    """Convolution + BatchNorm + ReLU6"""
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, 
                     groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    """Inverted Residual Block (MobileNetV2 building block)"""
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # Depthwise convolution
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # Pointwise projection (linear bottleneck)
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """
    MobileNetV2 for Bearing Fault Classification
    
    Args:
        num_classes: Number of output classes (default: 4)
        width_mult: Width multiplier for channel dimensions (default: 1.0)
        inverted_residual_setting: Network architecture specification
        round_nearest: Round channels to nearest multiple of this number (default: 8)
    """
    def __init__(self, num_classes=4, width_mult=1.0, 
                 inverted_residual_setting=None, round_nearest=8):
        super(MobileNetV2, self).__init__()
        
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        # Default architecture if not specified
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                # t: expansion factor, c: output channels, n: number of blocks, s: stride
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # Validate architecture
        if len(inverted_residual_setting) == 0 or \
           len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                           "or a 4-element list, got {}".format(inverted_residual_setting))

        # Adjust channels based on width multiplier
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        
        # First convolution layer
        features = [ConvBNReLU(3, input_channel, stride=2)]
        
        # Build inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        
        # Last convolution layer
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        
        # Make it nn.Sequential
        self.features = nn.Sequential(*features)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # For feature extraction during distillation
        self.intermediate_features = {}

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x, return_features=False):
        """
        Forward pass
        
        Args:
            x: Input tensor
            return_features: If True, return both logits and intermediate features
        
        Returns:
            logits or (logits, features) if return_features=True
        """
        # Extract features
        x = self.features(x)
        
        # Global average pooling
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        features = torch.flatten(x, 1)
        
        # Classification
        logits = self.classifier(features)
        
        if return_features:
            self.intermediate_features['final'] = features
            return logits, self.intermediate_features
        
        return logits

    def get_parameter_count(self):
        """Return number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameter_count(self):
        """Return number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def mobilenet_v2(num_classes=4, width_mult=1.0, pretrained=False):
    """
    Constructs a MobileNetV2 architecture
    
    Args:
        num_classes: Number of classes (default: 4)
        width_mult: Width multiplier (default: 1.0)
        pretrained: If True, loads ImageNet pretrained weights (default: False)
    """
    model = MobileNetV2(num_classes=num_classes, width_mult=width_mult)
    
    if pretrained:
        # Load ImageNet pretrained weights if needed
        # Note: Would need to adjust final layer for num_classes
        print("Warning: Pretrained weights not implemented yet")
    
    return model


if __name__ == '__main__':
    # Test the model
    model = mobilenet_v2(num_classes=4)
    print(f"MobileNetV2 Parameters: {model.get_parameter_count():,}")
    print(f"Trainable Parameters: {model.get_trainable_parameter_count():,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Test with feature extraction
    output, features = model(x, return_features=True)
    print(f"Features shape: {features['final'].shape}")

