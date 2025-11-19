"""
Visualize model predictions on CIFAR-10 images
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from torchvision import datasets, transforms
from src.models.wrn import wrn28x10
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


def load_model(checkpoint_path):
    """Load trained model"""
    model = wrn28x10(num_classes=10)
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()
    model.eval()
    return model


def predict_batch(model, images):
    """Get predictions for a batch of images"""
    with torch.no_grad():
        images = images.cuda()
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
    return predictions.cpu(), confidences.cpu()


def visualize_predictions(model, num_samples=20, save_path='outputs/predictions.png'):
    """Visualize model predictions with bounding boxes and labels"""
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    class_names = get_class_names()
    
    # Randomly select samples
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    # Get images and true labels
    images = []
    true_labels = []
    for idx in indices:
        img, label = test_dataset[idx]
        images.append(img)
        true_labels.append(label)
    
    images = torch.stack(images)
    
    # Get predictions
    pred_labels, confidences = predict_batch(model, images)
    
    # Create figure
    rows = 4
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
    fig.suptitle('Model Predictions on CIFAR-10 Test Set', fontsize=20, fontweight='bold', y=0.995)
    
    for idx, (ax, img, true_label, pred_label, conf) in enumerate(zip(
            axes.flat, images, true_labels, pred_labels, confidences)):
        
        # Denormalize and convert to numpy
        img_display = denormalize(img, mean, std)
        img_display = img_display.permute(1, 2, 0).numpy()
        
        # Display image
        ax.imshow(img_display)
        
        # Check if prediction is correct
        is_correct = (true_label == pred_label.item())
        
        # Create bounding box around the image
        rect = patches.Rectangle(
            (0, 0), 31, 31,
            linewidth=3,
            edgecolor='green' if is_correct else 'red',
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add prediction label with confidence
        pred_text = f"Pred: {class_names[pred_label]}\n({conf*100:.1f}%)"
        true_text = f"True: {class_names[true_label]}"
        
        # Background box for text
        ax.text(0.5, -0.15, pred_text, 
                transform=ax.transAxes,
                fontsize=10,
                ha='center',
                bbox=dict(boxstyle='round,pad=0.5', 
                         facecolor='lightgreen' if is_correct else 'lightcoral',
                         alpha=0.8, edgecolor='black', linewidth=1.5))
        
        ax.text(0.5, 1.05, true_text,
                transform=ax.transAxes,
                fontsize=10,
                ha='center',
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='lightblue',
                         alpha=0.8, edgecolor='black', linewidth=1))
        
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def visualize_correct_and_wrong(model, save_path='outputs/correct_vs_wrong.png'):
    """Show correct predictions vs wrong predictions"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    class_names = get_class_names()
    
    correct_samples = []
    wrong_samples = []
    
    # Find correct and wrong predictions
    for idx in range(len(test_dataset)):
        if len(correct_samples) >= 10 and len(wrong_samples) >= 10:
            break
            
        img, true_label = test_dataset[idx]
        
        # Get prediction
        pred_label, conf = predict_batch(model, img.unsqueeze(0))
        pred_label = pred_label.item()
        conf = conf.item()
        
        if pred_label == true_label and len(correct_samples) < 10:
            correct_samples.append((img, true_label, pred_label, conf))
        elif pred_label != true_label and len(wrong_samples) < 10:
            wrong_samples.append((img, true_label, pred_label, conf))
    
    # Create figure
    fig, axes = plt.subplots(2, 10, figsize=(25, 5))
    fig.suptitle('Correct Predictions (Top) vs Wrong Predictions (Bottom)', 
                 fontsize=18, fontweight='bold', y=1.02)
    
    # Plot correct predictions
    for idx, (img, true_label, pred_label, conf) in enumerate(correct_samples):
        ax = axes[0, idx]
        img_display = denormalize(img, mean, std).permute(1, 2, 0).numpy()
        ax.imshow(img_display)
        
        rect = patches.Rectangle((0, 0), 31, 31, linewidth=3, 
                                edgecolor='green', facecolor='none')
        ax.add_patch(rect)
        
        ax.text(0.5, -0.1, f"{class_names[pred_label]}\n{conf*100:.1f}%",
                transform=ax.transAxes, fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax.axis('off')
    
    # Plot wrong predictions
    for idx, (img, true_label, pred_label, conf) in enumerate(wrong_samples):
        ax = axes[1, idx]
        img_display = denormalize(img, mean, std).permute(1, 2, 0).numpy()
        ax.imshow(img_display)
        
        rect = patches.Rectangle((0, 0), 31, 31, linewidth=3,
                                edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        ax.text(0.5, 1.15, f"True: {class_names[true_label]}",
                transform=ax.transAxes, fontsize=8, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax.text(0.5, -0.1, f"Pred: {class_names[pred_label]}\n{conf*100:.1f}%",
                transform=ax.transAxes, fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def visualize_confidence_levels(model, save_path='outputs/confidence_levels.png'):
    """Visualize predictions with different confidence levels"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    class_names = get_class_names()
    
    # Find samples with different confidence levels
    high_conf = []  # > 99%
    medium_conf = []  # 90-95%
    low_conf = []  # < 85%
    
    for idx in range(len(test_dataset)):
        if len(high_conf) >= 8 and len(medium_conf) >= 8 and len(low_conf) >= 8:
            break
        
        img, true_label = test_dataset[idx]
        pred_label, conf = predict_batch(model, img.unsqueeze(0))
        pred_label = pred_label.item()
        conf = conf.item()
        
        if conf > 0.99 and len(high_conf) < 8:
            high_conf.append((img, true_label, pred_label, conf))
        elif 0.90 <= conf <= 0.95 and len(medium_conf) < 8:
            medium_conf.append((img, true_label, pred_label, conf))
        elif conf < 0.85 and len(low_conf) < 8:
            low_conf.append((img, true_label, pred_label, conf))
    
    # Create figure
    fig, axes = plt.subplots(3, 8, figsize=(20, 8))
    fig.suptitle('Predictions by Confidence Level', fontsize=18, fontweight='bold')
    
    titles = ['High Confidence (>99%)', 'Medium Confidence (90-95%)', 'Low Confidence (<85%)']
    colors = ['darkgreen', 'orange', 'darkred']
    
    for row_idx, (samples, title, color) in enumerate(zip([high_conf, medium_conf, low_conf], 
                                                           titles, colors)):
        for col_idx, (img, true_label, pred_label, conf) in enumerate(samples):
            ax = axes[row_idx, col_idx]
            img_display = denormalize(img, mean, std).permute(1, 2, 0).numpy()
            ax.imshow(img_display)
            
            is_correct = (true_label == pred_label)
            rect = patches.Rectangle((0, 0), 31, 31, linewidth=2.5,
                                    edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            status = "✓" if is_correct else "✗"
            ax.text(0.5, -0.12, f"{status} {class_names[pred_label]}\n{conf*100:.1f}%",
                    transform=ax.transAxes, fontsize=9, ha='center',
                    bbox=dict(boxstyle='round', 
                             facecolor='lightgreen' if is_correct else 'lightcoral',
                             alpha=0.8))
            
            if col_idx == 0:
                ax.text(-0.3, 0.5, title, transform=ax.transAxes,
                       fontsize=12, fontweight='bold', rotation=90,
                       va='center', ha='center')
            
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    import os
    os.makedirs('outputs', exist_ok=True)
    
    print("\n" + "="*70)
    print("Model Prediction Visualization")
    print("="*70 + "\n")
    
    checkpoint_path = 'runs/wrn28x10_ra_mixup/best9694.ckpt'
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠ Checkpoint not found: {checkpoint_path}")
        return
    
    print("Loading model...")
    model = load_model(checkpoint_path)
    print("✓ Model loaded successfully\n")
    
    print("1. Generating random predictions visualization...")
    visualize_predictions(model, num_samples=20)
    
    print("\n2. Generating correct vs wrong predictions comparison...")
    visualize_correct_and_wrong(model)
    
    print("\n3. Generating confidence level analysis...")
    visualize_confidence_levels(model)
    
    print("\n" + "="*70)
    print("✓ All prediction visualizations saved to 'outputs/' directory")
    print("="*70 + "\n")
    
    print("Generated files:")
    print("  - outputs/predictions.png (20 random predictions with boxes)")
    print("  - outputs/correct_vs_wrong.png (correct vs wrong predictions)")
    print("  - outputs/confidence_levels.png (predictions by confidence)")


if __name__ == '__main__':
    main()
