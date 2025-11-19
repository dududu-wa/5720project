"""
Implement Test-Time Augmentation (TTA) for improved accuracy
"""
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms

from src.models.wrn import wrn28x10, wrn40x4
from src.dataset import get_class_names


def load_model(checkpoint_path, model_type='wrn28x10'):
    """Load trained model"""
    if model_type == 'wrn28x10':
        model = wrn28x10(num_classes=10)
    elif model_type == 'wrn40x4':
        model = wrn40x4(num_classes=10)
    
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()
    model.eval()
    return model


def get_tta_transforms():
    """Get Test-Time Augmentation transforms"""
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    
    transforms_list = []
    
    # Original
    transforms_list.append(transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]))
    
    # Horizontal flip
    transforms_list.append(transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]))
    
    # Small crops
    for _ in range(3):
        transforms_list.append(transforms.Compose([
            transforms.RandomCrop(32, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]))
    
    return transforms_list


def evaluate_with_tta(model, test_loader_base, num_augmentations=5):
    """Evaluate model with Test-Time Augmentation"""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader_base, desc='TTA Evaluation'):
            batch_preds = []
            
            # Apply multiple augmentations
            for aug_idx in range(num_augmentations):
                if aug_idx == 0:
                    # Original
                    aug_images = images
                elif aug_idx == 1:
                    # Horizontal flip
                    aug_images = torch.flip(images, dims=[3])
                else:
                    # Random crop
                    pad = 2
                    padded = F.pad(images, (pad, pad, pad, pad), mode='reflect')
                    h, w = images.shape[2:]
                    i = torch.randint(0, 2*pad + 1, (1,)).item()
                    j = torch.randint(0, 2*pad + 1, (1,)).item()
                    aug_images = padded[:, :, i:i+h, j:j+w]
                
                aug_images = aug_images.cuda()
                outputs = model(aug_images)
                probs = F.softmax(outputs, dim=1)
                batch_preds.append(probs.cpu())
            
            # Average predictions
            avg_probs = torch.stack(batch_preds).mean(dim=0)
            preds = avg_probs.argmax(dim=1)
            
            all_preds.extend(preds.numpy())
            all_targets.extend(targets.numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    accuracy = 100.0 * np.sum(all_preds == all_targets) / len(all_targets)
    return accuracy, all_preds, all_targets


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate with TTA')
    parser.add_argument('--ckpt', required=True, type=str)
    parser.add_argument('--model', default='wrn28x10', type=str)
    parser.add_argument('--data_root', default='./data', type=str)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--num_aug', default=5, type=int, help='Number of augmentations')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("Test-Time Augmentation Evaluation")
    print("="*70 + "\n")
    
    # Load model
    print(f"Loading model from {args.ckpt}")
    model = load_model(args.ckpt, args.model)
    
    # Load test dataset
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_dataset = datasets.CIFAR10(
        root=args.data_root,
        train=False,
        download=False,
        transform=test_transform
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Evaluate with TTA
    print(f"\nEvaluating with {args.num_aug} augmentations per image...")
    tta_accuracy, preds, targets = evaluate_with_tta(model, test_loader, args.num_aug)
    
    # Compare with standard evaluation
    print("\n" + "="*70)
    print("Results:")
    print("="*70)
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.cuda()
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.cpu().eq(labels).sum().item()
    
    standard_accuracy = 100.0 * correct / total
    
    print(f"Standard Accuracy:     {standard_accuracy:.2f}%")
    print(f"TTA Accuracy:          {tta_accuracy:.2f}%")
    print(f"Improvement:           +{tta_accuracy - standard_accuracy:.2f}%")
    print("="*70 + "\n")
    
    print(f"✓ TTA provides {tta_accuracy - standard_accuracy:.2f}% improvement!")
    print(f"✓ Final accuracy with TTA: {tta_accuracy:.2f}%\n")


if __name__ == '__main__':
    main()
