"""
评估训练好的模型，包括基础准确率和TTA准确率
"""
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
from src.models.wrn import WideResNet

def load_model_with_ema(checkpoint_path, device):
    """加载模型并优先使用EMA权重"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = WideResNet(depth=28, widen_factor=10, num_classes=10)
    
    # 先加载标准权重以获取BN的running stats
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 然后用EMA权重覆盖可学习参数
    if 'ema_shadow' in checkpoint:
        print(f"Loading EMA weights from {checkpoint_path}")
        model_dict = model.state_dict()
        ema_dict = checkpoint['ema_shadow']
        
        # 只更新EMA中存在的参数（排除BN的running stats）
        for key in ema_dict:
            if key in model_dict:
                model_dict[key] = ema_dict[key]
        
        model.load_state_dict(model_dict)
        weight_type = "EMA (with BN stats from standard)"
    else:
        print(f"Loading standard weights from {checkpoint_path}")
        weight_type = "Standard"
    
    model.to(device)
    model.eval()
    
    # 显示checkpoint信息
    if 'best_acc' in checkpoint:
        print(f"  Checkpoint best accuracy: {checkpoint['best_acc']:.2f}%")
    if 'epoch' in checkpoint:
        print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    print(f"  Weight type: {weight_type}")
    
    return model

def get_test_loader(batch_size=100):
    """获取测试数据集"""
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=False,
        transform=test_transform
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return test_loader

def evaluate_base(model, test_loader, device):
    """基础评估（单次预测）"""
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Base Evaluation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy

def evaluate_tta(model, test_loader, device, num_augmentations=5):
    """TTA评估（多次预测集成）"""
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    
    # 定义TTA变换
    tta_transforms = [
        # 1. 原始图像
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        # 2. 水平翻转
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        # 3-5. 三次随机裁剪
        transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    ]
    
    # 获取原始数据集
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=False)
    
    all_predictions = []
    all_labels = []
    
    # 对每种变换进行评估
    for aug_idx, transform in enumerate(tta_transforms[:num_augmentations]):
        print(f"  TTA {aug_idx+1}/{num_augmentations}")
        
        # 应用当前变换
        test_dataset.transform = transform
        loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=100,
            shuffle=False,
            num_workers=2
        )
        
        batch_predictions = []
        batch_labels = []
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                batch_predictions.append(probs.cpu())
                batch_labels.append(labels)
        
        all_predictions.append(torch.cat(batch_predictions))
        if aug_idx == 0:
            all_labels = torch.cat(batch_labels)
    
    # 平均所有预测
    avg_predictions = torch.stack(all_predictions).mean(dim=0)
    final_predictions = avg_predictions.argmax(dim=1)
    
    # 计算准确率
    correct = (final_predictions == all_labels).sum().item()
    accuracy = 100.0 * correct / len(all_labels)
    
    return accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # 模型配置
    models = [
        {
            'name': 'Baseline (200 epochs, Mixup)',
            'checkpoint': 'runs/wrn28x10_ra_mixup/best9694.ckpt',
            'config': 'Mixup + RandAugment(N=2, M=9)'
        },
        {
            'name': 'Improved (300 epochs, CutMix)',
            'checkpoint': 'runs/wrn28x10_cutmix/best9716.ckpt',
            'config': 'CutMix + RandAugment(N=2, M=10)'
        }
    ]
    
    # 获取测试数据
    test_loader = get_test_loader()
    
    results = []
    
    for model_info in models:
        print("=" * 80)
        print(f"Evaluating: {model_info['name']}")
        print(f"Configuration: {model_info['config']}")
        print("-" * 80)
        
        # 加载模型
        model = load_model_with_ema(model_info['checkpoint'], device)
        
        # 基础评估
        print("\n[1/2] Base Evaluation (single prediction)...")
        base_acc = evaluate_base(model, test_loader, device)
        print(f"  Base Accuracy: {base_acc:.2f}%")
        
        # TTA评估
        print("\n[2/2] TTA Evaluation (5-way ensemble)...")
        tta_acc = evaluate_tta(model, test_loader, device, num_augmentations=5)
        print(f"  TTA Accuracy: {tta_acc:.2f}%")
        
        improvement = tta_acc - base_acc
        print(f"  Improvement: {improvement:+.2f}%")
        
        results.append({
            'name': model_info['name'],
            'config': model_info['config'],
            'base_acc': base_acc,
            'tta_acc': tta_acc,
            'improvement': improvement
        })
        
        print()
    
    # 生成报告
    print("\n" + "=" * 80)
    print("EVALUATION REPORT")
    print("=" * 80)
    print()
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['name']}")
        print(f"   Configuration: {result['config']}")
        print(f"   Base Accuracy:  {result['base_acc']:.2f}%")
        print(f"   TTA Accuracy:   {result['tta_acc']:.2f}%")
        print(f"   Improvement:    {result['improvement']:+.2f}%")
        print()
    
    # 对比分析
    print("-" * 80)
    print("COMPARISON ANALYSIS")
    print("-" * 80)
    base_diff = results[1]['base_acc'] - results[0]['base_acc']
    tta_diff = results[1]['tta_acc'] - results[0]['tta_acc']
    
    print(f"Base Accuracy Improvement (Improved vs Baseline): {base_diff:+.2f}%")
    print(f"TTA Accuracy Improvement (Improved vs Baseline):  {tta_diff:+.2f}%")
    print()
    print(f"Baseline TTA Gain: {results[0]['improvement']:+.2f}%")
    print(f"Improved TTA Gain: {results[1]['improvement']:+.2f}%")
    print()
    
    # 保存结果到文件
    with open('evaluation_results.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CIFAR-10 MODEL EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"{i}. {result['name']}\n")
            f.write(f"   Configuration: {result['config']}\n")
            f.write(f"   Base Accuracy:  {result['base_acc']:.2f}%\n")
            f.write(f"   TTA Accuracy:   {result['tta_acc']:.2f}%\n")
            f.write(f"   Improvement:    {result['improvement']:+.2f}%\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("COMPARISON ANALYSIS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Base Accuracy Improvement (Improved vs Baseline): {base_diff:+.2f}%\n")
        f.write(f"TTA Accuracy Improvement (Improved vs Baseline):  {tta_diff:+.2f}%\n\n")
        f.write(f"Baseline TTA Gain: {results[0]['improvement']:+.2f}%\n")
        f.write(f"Improved TTA Gain: {results[1]['improvement']:+.2f}%\n")
    
    print("Results saved to 'evaluation_results.txt'")
    print("=" * 80)

if __name__ == '__main__':
    main()
