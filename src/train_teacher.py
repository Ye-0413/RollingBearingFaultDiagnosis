"""
Teacher Model Training Script

Trains SE-ResNet152 teacher model on bearing fault data.
Saves checkpoints and can export soft labels for offline distillation.
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
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_factory import create_model
from src.data.dataset_loader import BearingDataset


class TeacherTrainer:
    """Trainer for teacher model"""
    
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
        
        # Initialize model, data, optimizer
        self.model = self._create_model()
        self.train_loader, self.val_loader = self._create_dataloaders()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = nn.CrossEntropyLoss()
        
        # Tracking
        self.best_val_acc = 0.0
        self.train_history = []
        self.val_history = []
    
    def save_config(self):
        """Save training configuration"""
        config = vars(self.args)
        config_path = self.output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Saved config to {config_path}")
    
    def _create_model(self):
        """Create teacher model"""
        print("\n" + "=" * 80)
        print("CREATING TEACHER MODEL")
        print("=" * 80)
        
        model = create_model(
            model_name='se_resnet152',
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
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.args.epochs} [Train]')
        
        for images, labels, _ in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
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
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
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
        print(f"Epochs: {self.args.epochs}")
        print(f"Batch size: {self.args.batch_size}")
        print(f"Learning rate: {self.args.lr}")
        
        for epoch in range(1, self.args.epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_history.append({'epoch': epoch, 'loss': train_loss, 'acc': train_acc})
            
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
        print("TRAINING COMPLETED")
        print("=" * 80)
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Teacher Model (SE-ResNet152)')
    
    # Data
    parser.add_argument('--train-data', type=str, required=True,
                        help='Path to training data annotation file')
    parser.add_argument('--val-data', type=str, required=True,
                        help='Path to validation data annotation file')
    parser.add_argument('--num-classes', type=int, default=4,
                        help='Number of classes')
    
    # Model
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights')
    parser.add_argument('--pretrained-path', type=str, default=None,
                        help='Path to pretrained checkpoint')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
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
    trainer = TeacherTrainer(args)
    trainer.train()

