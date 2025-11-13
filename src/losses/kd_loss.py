"""
Knowledge Distillation Loss Functions

Implements various KD loss functions:
1. Standard KD Loss (Hinton et al., 2015)
2. Feature-based Distillation Loss
3. Attention Transfer Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):
    """
    Standard Knowledge Distillation Loss
    
    Combines hard label loss (cross-entropy) with soft label loss (KL divergence).
    
    L_KD = α * L_CE(y, student_logits) + (1-α) * T^2 * KL(teacher_soft || student_soft)
    
    Args:
        temperature: Temperature for softening probability distributions (default: 4.0)
        alpha: Weight for hard label loss (default: 0.3)
                - alpha=1.0: Only hard labels (standard training)
                - alpha=0.0: Only soft labels (pure distillation)
    
    Reference:
        "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015)
    """
    
    def __init__(self, temperature=4.0, alpha=0.3):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, labels):
        """
        Compute KD loss
        
        Args:
            student_logits: Raw logits from student model [B, num_classes]
            teacher_logits: Raw logits from teacher model [B, num_classes]
            labels: Ground truth labels [B]
        
        Returns:
            loss: Combined distillation loss
            loss_dict: Dictionary with individual loss components
        """
        # Hard label loss (standard cross-entropy)
        hard_loss = self.ce_loss(student_logits, labels)
        
        # Soft label loss (KL divergence with temperature scaling)
        # Temperature makes the probability distributions softer
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        soft_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Combine losses
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'hard_loss': hard_loss.item(),
            'soft_loss': soft_loss.item(),
            'alpha': self.alpha,
            'temperature': self.temperature
        }
        
        return total_loss, loss_dict


class FeatureDistillationLoss(nn.Module):
    """
    Feature-based Distillation Loss
    
    Matches intermediate feature representations between teacher and student.
    Uses L2 distance to align feature maps.
    
    Args:
        feature_weight: Weight for feature loss (default: 1.0)
        normalize: Whether to normalize features before matching (default: True)
    
    Reference:
        "FitNets: Hints for Thin Deep Nets" (Romero et al., 2015)
    """
    
    def __init__(self, feature_weight=1.0, normalize=True):
        super(FeatureDistillationLoss, self).__init__()
        self.feature_weight = feature_weight
        self.normalize = normalize
    
    def forward(self, student_features, teacher_features):
        """
        Compute feature distillation loss
        
        Args:
            student_features: Feature tensor from student [B, C_s, H, W] or [B, C_s]
            teacher_features: Feature tensor from teacher [B, C_t, H, W] or [B, C_t]
        
        Returns:
            loss: Feature matching loss
        """
        # Normalize features if specified
        if self.normalize:
            student_features = F.normalize(student_features, p=2, dim=1)
            teacher_features = F.normalize(teacher_features, p=2, dim=1)
        
        # If channels don't match, need to add projection layer (handled externally)
        # Here we assume features are already aligned
        if student_features.shape != teacher_features.shape:
            # For spatial features, use adaptive pooling
            if len(student_features.shape) == 4:
                if student_features.shape[2:] != teacher_features.shape[2:]:
                    student_features = F.adaptive_avg_pool2d(
                        student_features, 
                        teacher_features.shape[2:]
                    )
        
        # L2 loss between features
        loss = F.mse_loss(student_features, teacher_features)
        
        return loss * self.feature_weight


class AttentionTransferLoss(nn.Module):
    """
    Attention Transfer Loss
    
    Matches attention maps (spatial importance) between teacher and student.
    
    Args:
        beta: Weight for attention transfer loss (default: 1000.0)
    
    Reference:
        "Paying More Attention to Attention" (Zagoruyko & Komodakis, 2017)
    """
    
    def __init__(self, beta=1000.0):
        super(AttentionTransferLoss, self).__init__()
        self.beta = beta
    
    def compute_attention_map(self, feature_map):
        """
        Compute attention map from feature map
        
        Args:
            feature_map: Feature tensor [B, C, H, W]
        
        Returns:
            attention_map: Attention map [B, H, W]
        """
        # Sum of squared activations across channels
        attention = torch.sum(feature_map ** 2, dim=1)  # [B, H, W]
        
        # Normalize
        batch_size = attention.shape[0]
        attention = attention.view(batch_size, -1)
        attention = F.normalize(attention, p=2, dim=1)
        attention = attention.view(feature_map.shape[0], feature_map.shape[2], feature_map.shape[3])
        
        return attention
    
    def forward(self, student_feature_maps, teacher_feature_maps):
        """
        Compute attention transfer loss
        
        Args:
            student_feature_maps: List of student feature maps
            teacher_feature_maps: List of teacher feature maps
        
        Returns:
            loss: Attention transfer loss
        """
        total_loss = 0
        num_stages = min(len(student_feature_maps), len(teacher_feature_maps))
        
        for i in range(num_stages):
            student_feat = student_feature_maps[i]
            teacher_feat = teacher_feature_maps[i]
            
            # Skip if not spatial features
            if len(student_feat.shape) != 4:
                continue
            
            # Compute attention maps
            student_attention = self.compute_attention_map(student_feat)
            teacher_attention = self.compute_attention_map(teacher_feat)
            
            # Match spatial dimensions if different
            if student_attention.shape != teacher_attention.shape:
                student_attention = F.interpolate(
                    student_attention.unsqueeze(1),
                    size=teacher_attention.shape[1:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
            
            # L2 loss between attention maps
            total_loss += F.mse_loss(student_attention, teacher_attention)
        
        return total_loss * self.beta / num_stages if num_stages > 0 else torch.tensor(0.0)


class CombinedDistillationLoss(nn.Module):
    """
    Combined Distillation Loss
    
    Combines multiple distillation objectives:
    - Standard KD loss (logits)
    - Feature distillation loss
    - Attention transfer loss
    
    Args:
        temperature: Temperature for KD loss (default: 4.0)
        alpha: Weight for hard labels (default: 0.3)
        feature_weight: Weight for feature loss (default: 0.5)
        attention_weight: Weight for attention loss (default: 0.1)
        use_feature_distillation: Whether to use feature distillation (default: False)
        use_attention_transfer: Whether to use attention transfer (default: False)
    """
    
    def __init__(
        self,
        temperature=4.0,
        alpha=0.3,
        feature_weight=0.5,
        attention_weight=0.1,
        use_feature_distillation=False,
        use_attention_transfer=False
    ):
        super(CombinedDistillationLoss, self).__init__()
        
        self.kd_loss = KnowledgeDistillationLoss(temperature, alpha)
        self.use_feature_distillation = use_feature_distillation
        self.use_attention_transfer = use_attention_transfer
        
        if use_feature_distillation:
            self.feature_loss = FeatureDistillationLoss(feature_weight)
        
        if use_attention_transfer:
            self.attention_loss = AttentionTransferLoss(attention_weight)
    
    def forward(
        self,
        student_logits,
        teacher_logits,
        labels,
        student_features=None,
        teacher_features=None
    ):
        """
        Compute combined distillation loss
        
        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            labels: Ground truth labels
            student_features: Dict of student intermediate features (optional)
            teacher_features: Dict of teacher intermediate features (optional)
        
        Returns:
            loss: Total combined loss
            loss_dict: Dictionary with all loss components
        """
        # Standard KD loss
        kd_loss, kd_dict = self.kd_loss(student_logits, teacher_logits, labels)
        total_loss = kd_loss
        loss_dict = kd_dict.copy()
        
        # Feature distillation loss
        if self.use_feature_distillation and student_features is not None and teacher_features is not None:
            # Match final features
            if 'final' in student_features and 'final' in teacher_features:
                feat_loss = self.feature_loss(
                    student_features['final'],
                    teacher_features['final']
                )
                total_loss += feat_loss
                loss_dict['feature_loss'] = feat_loss.item()
        
        # Attention transfer loss
        if self.use_attention_transfer and student_features is not None and teacher_features is not None:
            # Extract spatial feature maps
            student_maps = [v for k, v in student_features.items() if k.startswith('layer') or k.startswith('stage')]
            teacher_maps = [v for k, v in teacher_features.items() if k.startswith('layer') or k.startswith('stage')]
            
            if len(student_maps) > 0 and len(teacher_maps) > 0:
                attn_loss = self.attention_loss(student_maps, teacher_maps)
                total_loss += attn_loss
                loss_dict['attention_loss'] = attn_loss.item()
        
        loss_dict['total_combined_loss'] = total_loss.item()
        
        return total_loss, loss_dict


# Test the losses
if __name__ == '__main__':
    print("=" * 80)
    print("KNOWLEDGE DISTILLATION LOSS TEST")
    print("=" * 80)
    
    # Create dummy data
    batch_size = 8
    num_classes = 4
    
    student_logits = torch.randn(batch_size, num_classes)
    teacher_logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Test 1: Standard KD Loss
    print("\n" + "=" * 80)
    print("Test 1: Standard KD Loss")
    print("=" * 80)
    kd_loss_fn = KnowledgeDistillationLoss(temperature=4.0, alpha=0.3)
    loss, loss_dict = kd_loss_fn(student_logits, teacher_logits, labels)
    print(f"Loss: {loss.item():.4f}")
    for key, val in loss_dict.items():
        print(f"  {key}: {val}")
    
    # Test 2: Feature Distillation Loss
    print("\n" + "=" * 80)
    print("Test 2: Feature Distillation Loss")
    print("=" * 80)
    student_features = torch.randn(batch_size, 512)
    teacher_features = torch.randn(batch_size, 2048)
    # Need to match dimensions first (would use projection layer in practice)
    teacher_features_matched = teacher_features[:, :512]
    
    feat_loss_fn = FeatureDistillationLoss(feature_weight=1.0)
    feat_loss = feat_loss_fn(student_features, teacher_features_matched)
    print(f"Feature Loss: {feat_loss.item():.4f}")
    
    # Test 3: Attention Transfer Loss
    print("\n" + "=" * 80)
    print("Test 3: Attention Transfer Loss")
    print("=" * 80)
    student_maps = [
        torch.randn(batch_size, 128, 28, 28),
        torch.randn(batch_size, 256, 14, 14)
    ]
    teacher_maps = [
        torch.randn(batch_size, 512, 28, 28),
        torch.randn(batch_size, 1024, 14, 14)
    ]
    
    attn_loss_fn = AttentionTransferLoss(beta=1000.0)
    attn_loss = attn_loss_fn(student_maps, teacher_maps)
    print(f"Attention Loss: {attn_loss.item():.4f}")
    
    # Test 4: Combined Loss
    print("\n" + "=" * 80)
    print("Test 4: Combined Distillation Loss")
    print("=" * 80)
    combined_loss_fn = CombinedDistillationLoss(
        temperature=4.0,
        alpha=0.3,
        feature_weight=0.5,
        attention_weight=0.1,
        use_feature_distillation=True,
        use_attention_transfer=True
    )
    
    student_feats = {
        'final': student_features,
        'layer1': student_maps[0],
        'layer2': student_maps[1]
    }
    teacher_feats = {
        'final': teacher_features_matched,
        'layer1': teacher_maps[0],
        'layer2': teacher_maps[1]
    }
    
    combined_loss, combined_dict = combined_loss_fn(
        student_logits,
        teacher_logits,
        labels,
        student_feats,
        teacher_feats
    )
    print(f"Combined Loss: {combined_loss.item():.4f}")
    for key, val in combined_dict.items():
        print(f"  {key}: {val}")
    
    print("\n" + "=" * 80)
    print("✓ ALL LOSS FUNCTIONS TESTED SUCCESSFULLY")
    print("=" * 80)

