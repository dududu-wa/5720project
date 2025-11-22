# CIFAR-10 图像分类训练

基于 WideResNet-28-10 的 CIFAR-10 图像分类项目，使用 PyTorch 实现。

## 📁 项目结构

```
5720project/
├── data/                       # CIFAR-10 数据集
│   └── cifar-10-batches-py/
├── src/                        # 核心源代码
│   ├── models/                 # 模型定义
│   │   └── wrn.py             # WideResNet-28-10
│   ├── augment/               # 数据增强
│   │   ├── randaugment.py     # RandAugment
│   │   └── mixup_cutmix.py    # Mixup/CutMix
│   ├── dataset.py             # 数据集加载
│   ├── train.py               # 训练脚本
│   ├── eval.py                # 评估脚本
│   └── utils.py               # 工具函数
├── runs/                       # 训练输出
│   ├── wrn28x10_ra_mixup/     # 基线模型 (96.94%)
│   │   ├── best9694.ckpt
│   │   └── logs/
│   └── wrn28x10_cutmix/       # 改进模型 (97.16%)
│       ├── best9716.ckpt
│       └── logs/
├── requirements.txt            # 依赖包
├── run.ps1                     # 训练启动脚本
└── README.md                   # 本文件
```

## 🚀 快速开始

### 1. 环境配置

```powershell
# 创建虚拟环境
conda create -n cifar10 python=3.9
conda activate cifar10

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

数据集会在首次运行时自动下载到 `data/` 目录。

### 3. 训练模型

**基线配置**（Mixup + RandAugment）:
```powershell
python -m src.train --dataset cifar10 --data_root ./data --model wrn28x10 --epochs 200 --batch_size 128 --num_workers 4 --opt sgd --lr 0.1 --momentum 0.9 --wd 5e-4 --sched cosine --warmup 5 --randaugment N=2,M=9 --mixup 0.2 --cutmix 0.0 --label_smoothing 0.1 --ema 0.999 --amp --seed 42 --out runs/baseline
```

**改进配置**（CutMix + 更强增强）:
```powershell
python -m src.train --dataset cifar10 --data_root ./data --model wrn28x10 --epochs 300 --batch_size 128 --num_workers 4 --opt sgd --lr 0.1 --momentum 0.9 --wd 5e-4 --sched cosine --warmup 10 --randaugment N=2,M=10 --mixup 0.0 --cutmix 1.0 --label_smoothing 0.1 --ema 0.9995 --amp --seed 42 --out runs/improved
```

或使用提供的脚本：
```powershell
.\run.ps1
```

### 4. 评估模型

```powershell
python -m src.eval --dataset cifar10 --data_root ./data --model wrn28x10 --resume runs/wrn28x10_cutmix/best9716.ckpt
```

## 🎯 核心技术

### 模型架构
- **WideResNet-28-10**: 28层深度，10倍宽度因子
- **参数量**: ~36.5M
- **FLOPs**: ~5.2G

### 数据增强
- **RandAugment**: 自动数据增强策略
  - N=2: 应用2种增强
  - M=9-10: 增强强度
- **CutMix**: 区域混合正则化
  - α=1.0: 混合比例
- **标准增强**: 随机裁剪、水平翻转

### 训练策略
- **优化器**: SGD (momentum=0.9, wd=5e-4)
- **学习率**: 0.1，Cosine退火
- **Warmup**: 5-10 epochs
- **标签平滑**: ε=0.1
- **EMA**: 指数移动平均 (0.999-0.9995)
- **混合精度**: AMP加速训练

## 📊 性能结果

| 模型 | 配置 | 准确率 | Epochs | 训练时间 |
|-----|------|--------|--------|---------|
| Baseline | Mixup + RA(N=2,M=9) | 96.94% | 200 | ~13h |
| Improved | CutMix + RA(N=2,M=10) | 97.16% | 300 | ~20h |

*训练环境: NVIDIA RTX 3060 Laptop*

## 🔧 主要参数说明

| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `--epochs` | 训练轮数 | 200 |
| `--batch_size` | 批次大小 | 128 |
| `--lr` | 初始学习率 | 0.1 |
| `--randaugment` | RandAugment参数 | N=2,M=9 |
| `--mixup` | Mixup强度 | 0.0 |
| `--cutmix` | CutMix强度 | 1.0 |
| `--ema` | EMA衰减率 | 0.9995 |
| `--warmup` | 预热轮数 | 10 |

## 📝 依赖项

- Python >= 3.8
- PyTorch >= 1.10.0
- torchvision >= 0.11.0
- tensorboard
- numpy
- tqdm

详见 `requirements.txt`

## 🙏 致谢

- WideResNet: [Wide Residual Networks](https://arxiv.org/abs/1605.07146)
- RandAugment: [RandAugment: Practical automated data augmentation](https://arxiv.org/abs/1909.13719)
- CutMix: [CutMix: Regularization Strategy to Train Strong Classifiers](https://arxiv.org/abs/1905.04899)

---

*最后更新: 2025年11月*
