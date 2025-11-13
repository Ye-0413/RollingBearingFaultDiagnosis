#!/usr/bin/env python3
"""
Data Preparation Script for Knowledge Distillation

This script organizes the Training_data/ directory into proper train/test splits
and generates annotation files for each load condition.

Usage:
    python src/data/data_preparation.py --test-split 0.2 --seed 42
    python src/data/data_preparation.py --load 0 --test-split 0.2
    python src/data/data_preparation.py --create-combined

Author: Knowledge Distillation Framework
Date: June 2025
"""

import os
import sys
import argparse
import random
import shutil
from pathlib import Path
from collections import defaultdict
import json

# Add project root to path
sys.path.insert(0, os.getcwd())


class DataPreparation:
    """
    Comprehensive data preparation for bearing fault diagnosis
    
    Features:
    - Create train/test splits for each load condition
    - Generate annotation files
    - Validate dataset integrity
    - Create combined multi-load dataset
    - Generate dataset statistics
    """
    
    def __init__(self, data_root='Training_data', test_split=0.2, seed=42):
        """
        Initialize data preparation
        
        Args:
            data_root: Root directory containing load folders
            test_split: Fraction of data for testing (0.0 to 1.0)
            seed: Random seed for reproducibility
        """
        self.data_root = Path(data_root)
        self.test_split = test_split
        self.seed = seed
        random.seed(seed)
        
        # Class mapping (consistent across all loads)
        self.class_mapping = {
            'Ball': 0,
            'Ball_0': 0, 'Ball_1': 0, 'Ball_2': 0, 'Ball_3': 0,
            'OR': 1,
            'OR_0': 1, 'OR_1': 1, 'OR_2': 1, 'OR_3': 1,
            'IR': 2,
            'IR_0': 2, 'IR_1': 2, 'IR_2': 2, 'IR_3': 2,
            'Normal': 3,
            'Normal_0': 3, 'Normal_1': 3, 'Normal_2': 3, 'Normal_3': 3,
            'Cross_Ball': 0,
            'Cross_OR': 1,
            'Cross_IR': 2,
            'Cross_Normal': 3
        }
        
        # Load conditions
        self.load_conditions = ['load_0', 'load_1', 'load_2', 'load_3', 'Cross_load']
        
        self.stats = {
            'total_images': 0,
            'per_load': {},
            'per_class': defaultdict(int)
        }
    
    def validate_data_structure(self):
        """Validate that data directory structure is correct"""
        print("=" * 80)
        print("VALIDATING DATA STRUCTURE")
        print("=" * 80)
        
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root {self.data_root} does not exist!")
        
        all_valid = True
        for load in self.load_conditions:
            load_path = self.data_root / load
            if not load_path.exists():
                print(f"⚠️  WARNING: {load} directory not found")
                all_valid = False
                continue
            
            # List class directories
            class_dirs = [d for d in load_path.iterdir() if d.is_dir()]
            print(f"\n✓ {load}: Found {len(class_dirs)} class directories")
            
            for class_dir in class_dirs:
                images = list(class_dir.glob('*.png'))
                print(f"  - {class_dir.name}: {len(images)} images")
                
                if len(images) == 0:
                    print(f"    ⚠️  WARNING: No images found in {class_dir.name}")
                    all_valid = False
        
        if all_valid:
            print("\n✓ Data structure validation PASSED")
        else:
            print("\n⚠️  Data structure validation completed with warnings")
        
        return all_valid
    
    def create_train_test_split(self, load_name):
        """
        Create train/test split for a specific load condition
        
        Args:
            load_name: Name of load condition (e.g., 'load_0')
        
        Returns:
            train_files, test_files: Lists of (image_path, label) tuples
        """
        load_path = self.data_root / load_name
        if not load_path.exists():
            print(f"⚠️  Skipping {load_name}: directory not found")
            return [], []
        
        all_files = []
        
        # Collect all images
        for class_dir in load_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            if class_name not in self.class_mapping:
                print(f"⚠️  WARNING: Unknown class {class_name} in {load_name}")
                continue
            
            label = self.class_mapping[class_name]
            images = list(class_dir.glob('*.png'))
            
            for img_path in images:
                all_files.append((str(img_path.absolute()), label))
        
        # Shuffle and split
        random.shuffle(all_files)
        split_idx = int(len(all_files) * (1 - self.test_split))
        
        train_files = all_files[:split_idx]
        test_files = all_files[split_idx:]
        
        print(f"  {load_name}: {len(train_files)} train, {len(test_files)} test")
        
        return train_files, test_files
    
    def save_annotation_file(self, file_list, output_path):
        """
        Save annotation file in format: image_path label
        
        Args:
            file_list: List of (image_path, label) tuples
            output_path: Path to save annotation file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for img_path, label in file_list:
                f.write(f"{img_path} {label}\n")
        
        print(f"    Saved: {output_path} ({len(file_list)} samples)")
    
    def prepare_all_loads(self):
        """Prepare train/test splits for all load conditions"""
        print("\n" + "=" * 80)
        print("CREATING TRAIN/TEST SPLITS")
        print("=" * 80)
        print(f"Test split ratio: {self.test_split}")
        print(f"Random seed: {self.seed}\n")
        
        for load_name in self.load_conditions:
            print(f"Processing {load_name}...")
            train_files, test_files = self.create_train_test_split(load_name)
            
            if len(train_files) == 0:
                continue
            
            # Save train annotation
            train_path = self.data_root / load_name / 'train.txt'
            self.save_annotation_file(train_files, train_path)
            
            # Save test annotation
            test_path = self.data_root / load_name / 'test.txt'
            self.save_annotation_file(test_files, test_path)
            
            # Update statistics
            self.stats['per_load'][load_name] = {
                'train': len(train_files),
                'test': len(test_files),
                'total': len(train_files) + len(test_files)
            }
            self.stats['total_images'] += len(train_files) + len(test_files)
    
    def create_combined_dataset(self):
        """Create combined dataset from multiple loads"""
        print("\n" + "=" * 80)
        print("CREATING COMBINED MULTI-LOAD DATASET")
        print("=" * 80)
        
        # Combine load_0, load_1, load_2, load_3 (exclude Cross_load for testing)
        train_loads = ['load_0', 'load_1', 'load_2', 'load_3']
        
        combined_train = []
        combined_test = []
        
        for load_name in train_loads:
            train_file = self.data_root / load_name / 'train.txt'
            test_file = self.data_root / load_name / 'test.txt'
            
            if train_file.exists():
                with open(train_file, 'r') as f:
                    for line in f:
                        path, label = line.strip().split()
                        combined_train.append((path, int(label)))
            
            if test_file.exists():
                with open(test_file, 'r') as f:
                    for line in f:
                        path, label = line.strip().split()
                        combined_test.append((path, int(label)))
        
        # Shuffle combined datasets
        random.shuffle(combined_train)
        random.shuffle(combined_test)
        
        # Save combined datasets
        combined_dir = self.data_root / 'combined'
        combined_dir.mkdir(exist_ok=True)
        
        self.save_annotation_file(combined_train, combined_dir / 'train.txt')
        self.save_annotation_file(combined_test, combined_dir / 'test.txt')
        
        print(f"\n✓ Combined dataset created:")
        print(f"  - Train: {len(combined_train)} samples")
        print(f"  - Test: {len(combined_test)} samples")
    
    def create_class_annotation_file(self):
        """Create annotation file with class names"""
        class_names = {
            0: 'Ball',
            1: 'OR',
            2: 'IR',
            3: 'Normal'
        }
        
        anno_path = self.data_root / 'class_annotations.txt'
        with open(anno_path, 'w') as f:
            for label, name in sorted(class_names.items()):
                f.write(f"{name} {label}\n")
        
        print(f"\n✓ Class annotation file saved: {anno_path}")
    
    def generate_statistics_report(self):
        """Generate comprehensive dataset statistics"""
        print("\n" + "=" * 80)
        print("DATASET STATISTICS REPORT")
        print("=" * 80)
        
        print(f"\nTotal images: {self.stats['total_images']}")
        print(f"\nPer-load breakdown:")
        
        for load_name, counts in self.stats['per_load'].items():
            print(f"\n{load_name}:")
            print(f"  Train: {counts['train']} ({counts['train']/counts['total']*100:.1f}%)")
            print(f"  Test:  {counts['test']} ({counts['test']/counts['total']*100:.1f}%)")
            print(f"  Total: {counts['total']}")
        
        # Save statistics to JSON
        stats_path = self.data_root / 'dataset_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"\n✓ Statistics saved to: {stats_path}")
    
    def run(self, create_combined=True):
        """Run complete data preparation pipeline"""
        print("\n" + "=" * 80)
        print("KNOWLEDGE DISTILLATION DATA PREPARATION")
        print("=" * 80)
        
        # Step 1: Validate structure
        self.validate_data_structure()
        
        # Step 2: Create train/test splits
        self.prepare_all_loads()
        
        # Step 3: Create combined dataset
        if create_combined:
            self.create_combined_dataset()
        
        # Step 4: Create class annotation file
        self.create_class_annotation_file()
        
        # Step 5: Generate statistics
        self.generate_statistics_report()
        
        print("\n" + "=" * 80)
        print("✓ DATA PREPARATION COMPLETED SUCCESSFULLY")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Prepare bearing fault diagnosis data for knowledge distillation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        default='Training_data',
        help='Root directory containing training data'
    )
    
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.2,
        help='Fraction of data to use for testing (0.0 to 1.0)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--load',
        type=str,
        choices=['0', '1', '2', '3', 'cross', 'all'],
        default='all',
        help='Process specific load condition or all'
    )
    
    parser.add_argument(
        '--create-combined',
        action='store_true',
        default=True,
        help='Create combined multi-load dataset'
    )
    
    parser.add_argument(
        '--no-combined',
        dest='create_combined',
        action='store_false',
        help='Skip creating combined dataset'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.test_split < 0.0 or args.test_split > 1.0:
        raise ValueError("test_split must be between 0.0 and 1.0")
    
    # Create data preparation instance
    data_prep = DataPreparation(
        data_root=args.data_root,
        test_split=args.test_split,
        seed=args.seed
    )
    
    # Run preparation
    data_prep.run(create_combined=args.create_combined)


if __name__ == '__main__':
    main()

