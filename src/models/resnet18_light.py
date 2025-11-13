"""
ResNet18 Lightweight Implementation for Knowledge Distillation

Lightweight student model based on:
"Deep Residual Learning for Image Recognition" (He et al., 2016)

Parameters: ~11M (83% reduction vs SE-ResNet152)
"""

import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """Basic ResNet block for ResNet18/34"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    """
    ResNet18 for Bearing Fault Classification
    
    Args:
        num_classes: Number of output classes (default: 4)
        zero_init_residual: Zero-initialize the last BN in each residual branch
    """

    def __init__(self, num_classes=4, zero_init_residual=False):
        super(ResNet18, self).__init__()
        
        self.inplanes = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet18 architecture: [2, 2, 2, 2] blocks
        self.layer1 = self._make_layer(BasicBlock, 64, 2)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights(zero_init_residual)
        
        # For feature extraction during distillation
        self.intermediate_features = {}

    def _make_layer(self, block, planes, blocks, stride=1):
        """Create a residual layer"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self, zero_init_residual):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x, return_features=False):
        """
        Forward pass
        
        Args:
            x: Input tensor
            return_features: If True, return both logits and intermediate features
        
        Returns:
            logits or (logits, features) if return_features=True
        """
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual blocks
        x = self.layer1(x)
        if return_features:
            self.intermediate_features['layer1'] = x
        
        x = self.layer2(x)
        if return_features:
            self.intermediate_features['layer2'] = x
        
        x = self.layer3(x)
        if return_features:
            self.intermediate_features['layer3'] = x
        
        x = self.layer4(x)
        if return_features:
            self.intermediate_features['layer4'] = x

        # Global pooling and classification
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.fc(features)

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


def resnet18(num_classes=4, pretrained=False, **kwargs):
    """
    Constructs a ResNet-18 model
    
    Args:
        num_classes: Number of classes (default: 4)
        pretrained: If True, loads ImageNet pretrained weights (default: False)
    """
    model = ResNet18(num_classes=num_classes, **kwargs)
    
    if pretrained:
        # Load ImageNet pretrained weights if needed
        # Note: Would need to adjust final layer for num_classes
        print("Warning: Pretrained weights not implemented yet")
    
    return model


if __name__ == '__main__':
    # Test the model
    model = resnet18(num_classes=4)
    print(f"ResNet18 Parameters: {model.get_parameter_count():,}")
    print(f"Trainable Parameters: {model.get_trainable_parameter_count():,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Test with feature extraction
    output, features = model(x, return_features=True)
    print(f"Features shape: {features['final'].shape}")
    print(f"Number of intermediate features: {len(features)}")

