"""
Evaluation script for CIFAR-10
"""
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from src.models.wrn import wrn28x10, wrn40x4
from src.dataset import get_cifar10_dataloaders, get_class_names
from src.utils import set_seed, accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 Evaluation')
    parser.add_argument('--ckpt', required=True, type=str, help='Path to checkpoint')
    parser.add_argument('--data_root', default='./data', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--save_confusion_matrix', action='store_true', default=True)
    return parser.parse_args()


def evaluate(model, test_loader, class_names):
    """Evaluate the model and return predictions"""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs = inputs.cuda()
            outputs = model(inputs)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate accuracy
    acc = 100.0 * np.sum(all_preds == all_targets) / len(all_targets)
    
    return all_preds, all_targets, acc


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get class names
    class_names = get_class_names()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location='cuda', weights_only=False)
    
    # Get model architecture from checkpoint
    if 'args' in checkpoint:
        model_name = checkpoint['args'].model
    else:
        model_name = 'wrn28x10'
    
    # Create model
    print(f"Creating model: {model_name}")
    if model_name == 'wrn28x10':
        model = wrn28x10(num_classes=10)
    elif model_name == 'wrn40x4':
        model = wrn40x4(num_classes=10)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()
    
    # Create dataloader (without RandAugment for testing)
    print(f"Loading CIFAR-10 dataset from {args.data_root}")
    _, test_loader = get_cifar10_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_randaugment=False
    )
    
    # Evaluate
    print("\nEvaluating model...")
    preds, targets, acc = evaluate(model, test_loader, class_names)
    
    print(f"\nTest Accuracy: {acc:.2f}%")
    
    # Compute confusion matrix
    cm = confusion_matrix(targets, preds)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(targets, preds, target_names=class_names, digits=4))
    
    # Plot confusion matrix
    if args.save_confusion_matrix:
        output_dir = os.path.dirname(args.ckpt)
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(cm, class_names, cm_path)
    
    # Print per-class accuracy
    print("\nPer-class Accuracy:")
    for i, class_name in enumerate(class_names):
        class_acc = 100.0 * cm[i, i] / cm[i].sum()
        print(f"{class_name:12s}: {class_acc:.2f}%")
    
    # Find most confused pairs
    print("\nMost Confused Pairs:")
    np.fill_diagonal(cm, 0)
    confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm[i, j] > 0:
                confused_pairs.append((cm[i, j], class_names[i], class_names[j]))
    
    confused_pairs.sort(reverse=True)
    for count, true_class, pred_class in confused_pairs[:5]:
        print(f"{true_class} -> {pred_class}: {count} times")


if __name__ == '__main__':
    main()
