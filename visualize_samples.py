"""
Visualize CIFAR-10 samples and augmentations
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from src.augment.randaugment import RandAugment
from src.dataset import get_class_names

# CIFAR-10 normalization constants
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]


def denormalize(img, mean, std):
    """Denormalize image for visualization"""
    img = img.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(img, 0, 1)


def visualize_original_samples():
    """Visualize original CIFAR-10 samples"""
    # Load dataset without augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    class_names = get_class_names()
    
    # Get one sample from each class
    samples = []
    labels = []
    for class_idx in range(10):
        # Find first sample of this class
        for i in range(len(dataset)):
            _, label = dataset[i]
            if label == class_idx:
                img, _ = dataset[i]
                samples.append(img)
                labels.append(label)
                break
    
    # Plot
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('CIFAR-10 Sample Images (One per Class)', fontsize=16, fontweight='bold')
    
    for idx, (ax, img, label) in enumerate(zip(axes.flat, samples, labels)):
        img_display = denormalize(img, mean, std)
        img_display = img_display.permute(1, 2, 0).numpy()
        
        ax.imshow(img_display)
        ax.set_title(f'{class_names[label]}', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/cifar10_samples.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: outputs/cifar10_samples.png")
    plt.close()


def visualize_augmentations():
    """Visualize RandAugment effects"""
    # Load one image
    dataset = datasets.CIFAR10(root='./data', train=True, download=False)
    img_pil, label = dataset[0]  # Get PIL image
    class_names = get_class_names()
    
    # Create RandAugment
    ra = RandAugment(n=2, m=9)
    
    # Apply augmentation multiple times
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle(f'RandAugment Examples (Original: {class_names[label]})', 
                 fontsize=16, fontweight='bold')
    
    # First image is original
    axes[0, 0].imshow(img_pil)
    axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Apply augmentation 14 times
    for idx in range(1, 15):
        ax = axes[idx // 5, idx % 5]
        augmented = ra(img_pil.copy())
        ax.imshow(augmented)
        ax.set_title(f'Augmented {idx}', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/randaugment_examples.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: outputs/randaugment_examples.png")
    plt.close()


def visualize_training_progress():
    """Visualize training progress from checkpoints"""
    import os
    
    if not os.path.exists('runs/wrn28x10_ra_mixup/best.ckpt'):
        print("⚠ Training checkpoint not found. Please wait for training to complete.")
        return
    
    # Load checkpoint
    checkpoint = torch.load('runs/wrn28x10_ra_mixup/best.ckpt', map_location='cpu')
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    best_acc = checkpoint.get('best_acc', 0)
    epoch = checkpoint.get('epoch', 0)
    
    ax.text(0.5, 0.6, f'Best Accuracy: {best_acc:.2f}%', 
            ha='center', va='center', fontsize=24, fontweight='bold')
    ax.text(0.5, 0.4, f'Epoch: {epoch + 1}/200', 
            ha='center', va='center', fontsize=18)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Training Progress', fontsize=20, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('outputs/training_progress.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: outputs/training_progress.png")
    print(f"  Best Accuracy: {best_acc:.2f}% at Epoch {epoch + 1}")
    plt.close()


def visualize_batch():
    """Visualize a batch of training images"""
    from src.dataset import get_cifar10_dataloaders
    
    train_loader, _ = get_cifar10_dataloaders(
        data_root='./data',
        batch_size=16,
        num_workers=0,
        use_randaugment=True,
        ra_n=2,
        ra_m=9
    )
    
    # Get one batch
    images, labels = next(iter(train_loader))
    class_names = get_class_names()
    
    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('Training Batch with RandAugment', fontsize=16, fontweight='bold')
    
    for idx, (ax, img, label) in enumerate(zip(axes.flat, images, labels)):
        img_display = denormalize(img, mean, std)
        img_display = img_display.permute(1, 2, 0).numpy()
        
        ax.imshow(img_display)
        ax.set_title(f'{class_names[label]}', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/training_batch.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: outputs/training_batch.png")
    plt.close()


def main():
    import os
    os.makedirs('outputs', exist_ok=True)
    
    print("\n" + "="*60)
    print("CIFAR-10 Visualization")
    print("="*60 + "\n")
    
    print("1. Generating original samples (one per class)...")
    visualize_original_samples()
    
    print("\n2. Generating RandAugment examples...")
    visualize_augmentations()
    
    print("\n3. Generating training batch visualization...")
    visualize_batch()
    
    print("\n4. Checking training progress...")
    visualize_training_progress()
    
    print("\n" + "="*60)
    print("✓ All visualizations saved to 'outputs/' directory")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
