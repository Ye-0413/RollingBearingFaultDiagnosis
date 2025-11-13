"""
Student Model Training Script

Trains student models (MobileNetV2, ResNet18) with multiple modes:
1. Baseline: Standard supervised learning
2. Distillation: Knowledge distillation from teacher model (online or offline)
3. Quantization-Aware Training (QAT): Optional INT8 quantization after FP32 training

QAT Mode:
- Automatically triggered with --quantization-aware flag
- First trains FP32 model normally, then fine-tunes with quantization simulation
- Produces both FP32 and INT8 models for comparison
- INT8 model achieves ~4x size reduction with minimal accuracy loss
"""

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization as quant
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_factory import create_model
from src.data.dataset_loader import BearingDataset
from src.losses.kd_loss import KnowledgeDistillationLoss, CombinedDistillationLoss


class StudentTrainer:
    """Trainer for student models with optional knowledge distillation"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directories
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir = self.output_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)
        
        # Save configuration
        self.save_config()
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize teacher if distillation mode
        self.teacher = None
        if args.distillation_mode != 'none':
            self.teacher = self._load_teacher()
        
        # Data loaders
        self.train_loader, self.val_loader = self._create_dataloaders()
        
        # Optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Loss function
        self.criterion = self._create_criterion()
        
        # Tracking
        self.best_val_acc = 0.0
        self.train_history = []
        self.val_history = []
        
        # QAT state
        self.qat_mode = False
        self.fp32_model_saved = False
    
    def save_config(self):
        """Save training configuration"""
        config = vars(self.args)
        config_path = self.output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Saved config to {config_path}")
    
    def _create_model(self):
        """Create student model"""
        print("\n" + "=" * 80)
        print("CREATING STUDENT MODEL")
        print("=" * 80)
        
        model = create_model(
            model_name=self.args.model,
            num_classes=self.args.num_classes,
            pretrained=self.args.pretrained,
            pretrained_path=self.args.pretrained_path
        )
        
        model = model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {total_params * 4 / (1024**2):.2f} MB")
        
        return model
    
    def _load_teacher(self):
        """Load teacher model for distillation"""
        print("\n" + "=" * 80)
        print("LOADING TEACHER MODEL")
        print("=" * 80)
        
        if not self.args.teacher_checkpoint:
            raise ValueError("Teacher checkpoint required for distillation mode")
        
        # Create teacher model
        teacher = create_model(
            model_name='se_resnet152',
            num_classes=self.args.num_classes
        )
        
        # Load checkpoint
        checkpoint = torch.load(self.args.teacher_checkpoint, map_location='cpu', weights_only=True)
        
        if 'model_state_dict' in checkpoint:
            teacher.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            teacher.load_state_dict(checkpoint['state_dict'])
        else:
            teacher.load_state_dict(checkpoint)
        
        teacher = teacher.to(self.device)
        teacher.eval()
        
        # Freeze teacher parameters
        for param in teacher.parameters():
            param.requires_grad = False
        
        print(f"✓ Loaded teacher from {self.args.teacher_checkpoint}")
        
        return teacher
    
    def _create_dataloaders(self):
        """Create train and validation dataloaders"""
        print("\n" + "=" * 80)
        print("CREATING DATALOADERS")
        print("=" * 80)
        
        # Define transforms
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = BearingDataset(
            data_path=self.args.train_data,
            transform=train_transform
        )
        
        val_dataset = BearingDataset(
            data_path=self.args.val_data,
            transform=val_transform
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def _create_optimizer(self):
        """Create optimizer"""
        if self.args.optimizer == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.args.optimizer}")
        
        print(f"\n✓ Created {self.args.optimizer.upper()} optimizer (lr={self.args.lr})")
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.args.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.args.step_size,
                gamma=self.args.gamma
            )
        elif self.args.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.args.epochs
            )
        elif self.args.scheduler == 'multistep':
            milestones = [int(self.args.epochs * 0.5), int(self.args.epochs * 0.75)]
            scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=self.args.gamma
            )
        else:
            scheduler = None
        
        print(f"✓ Created {self.args.scheduler} scheduler")
        return scheduler
    
    def _create_criterion(self):
        """Create loss function based on mode"""
        if self.args.distillation_mode == 'none':
            # Standard cross-entropy for baseline
            criterion = nn.CrossEntropyLoss()
            print(f"\n✓ Using standard CrossEntropyLoss (Baseline mode)")
        else:
            # Knowledge distillation loss
            if self.args.use_feature_distillation or self.args.use_attention_transfer:
                criterion = CombinedDistillationLoss(
                    temperature=self.args.temperature,
                    alpha=self.args.alpha,
                    feature_weight=self.args.feature_weight,
                    attention_weight=self.args.attention_weight,
                    use_feature_distillation=self.args.use_feature_distillation,
                    use_attention_transfer=self.args.use_attention_transfer
                )
                print(f"\n✓ Using CombinedDistillationLoss")
                print(f"  Temperature: {self.args.temperature}")
                print(f"  Alpha: {self.args.alpha}")
                print(f"  Feature distillation: {self.args.use_feature_distillation}")
                print(f"  Attention transfer: {self.args.use_attention_transfer}")
            else:
                criterion = KnowledgeDistillationLoss(
                    temperature=self.args.temperature,
                    alpha=self.args.alpha
                )
                print(f"\n✓ Using KnowledgeDistillationLoss")
                print(f"  Temperature: {self.args.temperature}")
                print(f"  Alpha: {self.args.alpha}")
        
        return criterion
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        loss_components = {}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.args.epochs} [Train]')
        
        for images, labels, _ in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass - different for baseline vs distillation
            if self.args.distillation_mode == 'none':
                # Baseline: Standard training
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss_dict = {'loss': loss.item()}
            
            elif self.args.distillation_mode == 'online':
                # Online distillation: Teacher forward pass at training time
                with torch.no_grad():
                    if self.args.use_feature_distillation or self.args.use_attention_transfer:
                        teacher_logits, teacher_features = self.teacher(images, return_features=True)
                    else:
                        teacher_logits = self.teacher(images)
                        teacher_features = None
                
                # Student forward pass
                if self.args.use_feature_distillation or self.args.use_attention_transfer:
                    student_logits, student_features = self.model(images, return_features=True)
                    loss, loss_dict = self.criterion(
                        student_logits, teacher_logits, labels,
                        student_features, teacher_features
                    )
                else:
                    student_logits = self.model(images)
                    loss, loss_dict = self.criterion(student_logits, teacher_logits, labels)
                
                outputs = student_logits
            
            else:
                raise ValueError(f"Unknown distillation mode: {self.args.distillation_mode}")
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Accumulate loss components
            for key, val in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                if isinstance(val, (int, float)):
                    loss_components[key] += val * images.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        # Average loss components
        for key in loss_components:
            loss_components[key] /= total
        
        return epoch_loss, epoch_acc, loss_components
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch}/{self.args.epochs} [Val]')
            
            for images, labels, _ in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass (always standard for validation)
                outputs = self.model(images)
                loss = nn.functional.cross_entropy(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, all_predictions, all_labels
    
    def prepare_for_qat(self):
        """
        Prepare model for Quantization-Aware Training.
        Called after initial FP32 training is complete.
        """
        print("\n" + "=" * 80)
        print("PREPARING FOR QUANTIZATION-AWARE TRAINING")
        print("=" * 80)
        
        # Save FP32 model first
        if not self.fp32_model_saved:
            fp32_path = self.checkpoint_dir / 'best_model_fp32.pth'
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'best_val_acc': self.best_val_acc,
                'args': vars(self.args)
            }
            torch.save(checkpoint, fp32_path)
            print(f"✓ Saved FP32 model to {fp32_path}")
            self.fp32_model_saved = True
        
        # Set quantization backend
        torch.backends.quantized.engine = self.args.qat_backend
        print(f"✓ Set quantization backend: {self.args.qat_backend}")
        
        # Fuse modules (Conv+BN+ReLU)
        # Note: This is model-specific. For MobileNetV2/ResNet18, we'll use generic fusing
        self.model.eval()
        
        # Set quantization config
        self.model.qconfig = quant.get_default_qat_qconfig(self.args.qat_backend)
        print(f"✓ Set QAT qconfig")
        
        # Prepare for QAT (insert fake quantization modules)
        quant.prepare_qat(self.model, inplace=True)
        print("✓ Inserted fake quantization modules")
        
        # Switch to QAT mode
        self.qat_mode = True
        self.model.train()
        
        # Create new optimizer with lower learning rate for fine-tuning
        print(f"\n✓ Creating QAT optimizer with lr={self.args.qat_lr}")
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.qat_lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.qat_lr,
                weight_decay=self.args.weight_decay
            )
        
        # Reset scheduler for QAT phase
        if self.args.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.args.qat_num_epochs
            )
        else:
            self.scheduler = None
        
        print("=" * 80)
        print(f"QAT READY - Will fine-tune for {self.args.qat_num_epochs} epochs")
        print("=" * 80)
    
    def convert_to_quantized(self):
        """
        Convert QAT model to fully quantized INT8 model.
        Called after QAT fine-tuning is complete.
        """
        print("\n" + "=" * 80)
        print("CONVERTING TO QUANTIZED INT8 MODEL")
        print("=" * 80)
        
        self.model.eval()
        self.model = quant.convert(self.model, inplace=True)
        
        print("✓ Converted to INT8 quantized model")
        
        # Save quantized model
        quantized_path = self.checkpoint_dir / 'best_model_int8_qat.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_val_acc': self.best_val_acc,
            'args': vars(self.args),
            'quantization_method': 'qat',
            'backend': self.args.qat_backend
        }, quantized_path)
        print(f"✓ Saved INT8 model to {quantized_path}")
        
        print("=" * 80)
        print("QUANTIZATION COMPLETE")
        print("=" * 80)
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'args': vars(self.args)
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (acc: {self.best_val_acc:.2f}%)")
        
        # Keep only last N checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        if len(checkpoints) > self.args.keep_checkpoints:
            for old_ckpt in checkpoints[:-self.args.keep_checkpoints]:
                old_ckpt.unlink()
    
    def train(self):
        """Main training loop"""
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Model: {self.args.model}")
        print(f"Mode: {'Baseline' if self.args.distillation_mode == 'none' else f'Distillation ({self.args.distillation_mode})'}")
        print(f"Epochs: {self.args.epochs}")
        print(f"Batch size: {self.args.batch_size}")
        print(f"Learning rate: {self.args.lr}")
        
        for epoch in range(1, self.args.epochs + 1):
            # Train
            train_loss, train_acc, loss_components = self.train_epoch(epoch)
            self.train_history.append({
                'epoch': epoch,
                'loss': train_loss,
                'acc': train_acc,
                'loss_components': loss_components
            })
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate(epoch)
            self.val_history.append({'epoch': epoch, 'loss': val_loss, 'acc': val_acc})
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
            else:
                current_lr = self.args.lr
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{self.args.epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            if loss_components and self.args.distillation_mode != 'none':
                print("  Loss Components:")
                for key, val in loss_components.items():
                    if isinstance(val, (int, float)):
                        print(f"    {key}: {val:.4f}")
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            
            if epoch % self.args.save_freq == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        # Save training history
        history_path = self.log_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump({
                'train': self.train_history,
                'val': self.val_history
            }, f, indent=2)
        
        print("\n" + "=" * 80)
        print("FP32 TRAINING COMPLETED")
        print("=" * 80)
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        
        # Quantization-Aware Training (if enabled)
        if self.args.quantization_aware:
            print("\n" + "=" * 80)
            print("STARTING QUANTIZATION-AWARE TRAINING (QAT)")
            print("=" * 80)
            
            # Load best FP32 model before QAT
            best_checkpoint = torch.load(self.checkpoint_dir / 'best_model.pth')
            self.model.load_state_dict(best_checkpoint['model_state_dict'])
            print("✓ Loaded best FP32 model for QAT")
            
            # Prepare for QAT
            self.prepare_for_qat()
            
            # QAT fine-tuning loop
            qat_best_acc = 0.0
            for epoch in range(1, self.args.qat_num_epochs + 1):
                # Train
                train_loss, train_acc, loss_components = self.train_epoch(epoch)
                self.train_history.append({
                    'epoch': self.args.epochs + epoch,
                    'loss': train_loss,
                    'acc': train_acc,
                    'qat': True
                })
                
                # Validate
                val_loss, val_acc, val_preds, val_labels = self.validate(epoch)
                self.val_history.append({
                    'epoch': self.args.epochs + epoch,
                    'loss': val_loss,
                    'acc': val_acc,
                    'qat': True
                })
                
                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()
                    current_lr = self.optimizer.param_groups[0]['lr']
                else:
                    current_lr = self.args.qat_lr
                
                # Print epoch summary
                print(f"\nQAT Epoch {epoch}/{self.args.qat_num_epochs} Summary:")
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
                print(f"  Learning Rate: {current_lr:.6f}")
                
                # Track best QAT accuracy
                if val_acc > qat_best_acc:
                    qat_best_acc = val_acc
            
            # Update best accuracy if QAT improved it
            if qat_best_acc > self.best_val_acc:
                self.best_val_acc = qat_best_acc
            
            # Convert to INT8
            self.convert_to_quantized()
            
            # Final validation on INT8 model
            print("\n" + "=" * 80)
            print("FINAL VALIDATION ON INT8 MODEL")
            print("=" * 80)
            val_loss, val_acc, _, _ = self.validate(0)
            print(f"INT8 Model Validation Accuracy: {val_acc:.2f}%")
            print(f"FP32 vs INT8 Accuracy Degradation: {self.best_val_acc - val_acc:.2f}%")
            
            # Update history with INT8 results
            history_path = self.log_dir / 'training_history_with_qat.json'
            with open(history_path, 'w') as f:
                json.dump({
                    'train': self.train_history,
                    'val': self.val_history,
                    'qat_epochs': self.args.qat_num_epochs,
                    'int8_final_acc': val_acc
                }, f, indent=2)
            
            print("\n" + "=" * 80)
            print("QAT TRAINING COMPLETED")
            print("=" * 80)
            print(f"Best FP32 accuracy: {self.best_val_acc:.2f}%")
            print(f"Final INT8 accuracy: {val_acc:.2f}%")
            print(f"Quantized model saved to: {self.checkpoint_dir / 'best_model_int8_qat.pth'}")
        else:
            print("\nSkipping QAT (--quantization-aware not specified)")
            print("To enable QAT, add --quantization-aware flag")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Student Model (Baseline or Distillation)')
    
    # Data
    parser.add_argument('--train-data', type=str, required=True,
                        help='Path to training data annotation file')
    parser.add_argument('--val-data', type=str, required=True,
                        help='Path to validation data annotation file')
    parser.add_argument('--num-classes', type=int, default=4,
                        help='Number of classes')
    
    # Model
    parser.add_argument('--model', type=str, required=True,
                        choices=['mobilenetv2', 'resnet18'],
                        help='Student model architecture')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights')
    parser.add_argument('--pretrained-path', type=str, default=None,
                        help='Path to pretrained checkpoint')
    
    # Distillation
    parser.add_argument('--distillation-mode', type=str, default='none',
                        choices=['none', 'online', 'offline'],
                        help='Distillation mode (none=baseline)')
    parser.add_argument('--teacher-checkpoint', type=str, default=None,
                        help='Path to teacher checkpoint (required for distillation)')
    parser.add_argument('--temperature', type=float, default=4.0,
                        help='Temperature for knowledge distillation')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Weight for hard labels (1-alpha for soft labels)')
    parser.add_argument('--use-feature-distillation', action='store_true',
                        help='Use feature-based distillation')
    parser.add_argument('--feature-weight', type=float, default=0.5,
                        help='Weight for feature distillation loss')
    parser.add_argument('--use-attention-transfer', action='store_true',
                        help='Use attention transfer')
    parser.add_argument('--attention-weight', type=float, default=0.1,
                        help='Weight for attention transfer loss')
    
    # Quantization-Aware Training (QAT)
    parser.add_argument('--quantization-aware', action='store_true',
                        help='Enable quantization-aware training (INT8)')
    parser.add_argument('--qat-backend', type=str, default='fbgemm',
                        choices=['fbgemm', 'qnnpack'],
                        help='QAT backend (fbgemm for x86, qnnpack for ARM)')
    parser.add_argument('--qat-num-epochs', type=int, default=20,
                        help='Number of QAT fine-tuning epochs after regular training')
    parser.add_argument('--qat-lr', type=float, default=0.0001,
                        help='Learning rate for QAT fine-tuning (lower than regular)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam'],
                        help='Optimizer')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    
    # Scheduler
    parser.add_argument('--scheduler', type=str, default='step',
                        choices=['step', 'cosine', 'multistep', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--step-size', type=int, default=30,
                        help='Step size for StepLR')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Gamma for lr scheduler')
    
    # I/O
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--save-freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--keep-checkpoints', type=int, default=5,
                        help='Number of recent checkpoints to keep')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Create timestamp for output directory if not absolute path
    if not os.path.isabs(args.output_dir):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f"{args.output_dir}_{timestamp}"
    
    # Train
    trainer = StudentTrainer(args)
    trainer.train()

