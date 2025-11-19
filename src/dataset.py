"""
CIFAR-10 dataset loader with transforms
"""
import torch
from torchvision import datasets, transforms
from src.augment.randaugment import RandAugment


def get_cifar10_dataloaders(data_root='./data', batch_size=128, num_workers=4, 
                             use_randaugment=True, ra_n=2, ra_m=9):
    """
    Get CIFAR-10 train and test dataloaders
    
    Args:
        data_root: Root directory for dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        use_randaugment: Whether to use RandAugment
        ra_n: RandAugment N parameter
        ra_m: RandAugment M parameter
    
    Returns:
        train_loader, test_loader
    """
    # CIFAR-10 normalization constants
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    
    # Training transforms
    train_transform_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    
    if use_randaugment:
        train_transform_list.append(RandAugment(n=ra_n, m=ra_m))
    
    train_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_transform = transforms.Compose(train_transform_list)
    
    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def get_class_names():
    """Get CIFAR-10 class names"""
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck']
