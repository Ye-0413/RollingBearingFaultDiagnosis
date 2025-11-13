"""
Dataset Loader for Knowledge Distillation

Custom PyTorch Dataset for loading bearing fault images with support for:
- Train/test splits from annotation files
- Data augmentation
- Teacher logits loading for offline distillation
"""

import os
import sys
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Add project root to path
sys.path.insert(0, os.getcwd())


class BearingDataset(Dataset):
    """
    Bearing Fault Diagnosis Dataset
    
    Args:
        annotation_file: Path to annotation file (format: image_path label)
        transform: Optional transform to be applied on images
        teacher_logits_file: Optional path to pre-computed teacher logits
    """
    
    def __init__(self, annotation_file, transform=None, teacher_logits_file=None):
        self.annotation_file = annotation_file
        self.transform = transform
        self.teacher_logits = None
        
        # Load annotations
        self.samples = []
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    img_path, label = parts
                    self.samples.append((img_path, int(label)))
        
        # Load teacher logits if provided (for offline distillation)
        if teacher_logits_file and os.path.exists(teacher_logits_file):
            import pickle
            with open(teacher_logits_file, 'rb') as f:
                self.teacher_logits = pickle.load(f)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Return with teacher logits if available
        if self.teacher_logits and img_path in self.teacher_logits:
            teacher_logit = torch.tensor(
                self.teacher_logits[img_path]['logits'],
                dtype=torch.float32
            )
            return image, label, teacher_logit, img_path
        
        return image, label, img_path


def get_train_transforms(image_size=224):
    """Get training data transformations with augmentation"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_test_transforms(image_size=224):
    """Get test data transformations (no augmentation)"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

