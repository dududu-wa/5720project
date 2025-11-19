# CIFAR-10 Classification with WideResNet

This project implements a state-of-the-art CIFAR-10 image classifier using WideResNet-28-10 with advanced data augmentation techniques.

## Project Structure

```
cifar10-project/
├── src/
│   ├── models/
│   │   └── wrn.py              # WideResNet implementation
│   ├── augment/
│   │   ├── randaugment.py      # RandAugment implementation
│   │   └── mixup_cutmix.py     # Mixup and CutMix implementations
│   ├── dataset.py              # CIFAR-10 data loaders
│   ├── train.py                # Training script
│   ├── eval.py                 # Evaluation script
│   └── utils.py                # Utility functions
├── data/                        # CIFAR-10 dataset (auto-downloaded)
├── runs/                        # Training outputs and checkpoints
├── requirements.txt             # Python dependencies
├── run.ps1                      # PowerShell training script
└── README.md                    # This file
```

## Features

- **Model**: WideResNet-28-10 (depth=28, width=10)
- **Data Augmentation**: 
  - RandAugment (N=2, M=9)
  - Mixup (α=0.2)
  - RandomCrop with padding
  - RandomHorizontalFlip
- **Training Techniques**:
  - SGD optimizer with Nesterov momentum
  - Cosine annealing learning rate schedule with warmup
  - Label smoothing (0.1)
  - Exponential Moving Average (EMA, decay=0.999)
  - Mixed precision training (AMP)
- **Regularization**: Weight decay (5e-4)

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA support
- NVIDIA GPU (tested on RTX 3060 Laptop)
- Conda environment: `pythonProject1`

## Installation

1. Activate conda environment:
```powershell
conda activate pythonProject1
```

2. Install dependencies:
```powershell
pip install -r requirements.txt
```

## Quick Start Guide

### Option 1: Full Training (200 epochs, ~8-12 hours)

```powershell
.\run.ps1
```

### Option 2: Quick Test (5 epochs, ~10-15 minutes)

```powershell
.\test_run.ps1
```

### Option 3: Monitor Training Progress

```powershell
.\monitor.ps1
```

### Manual Training

```powershell
conda activate pythonProject1

python -m src.train `
  --dataset cifar10 `
  --data_root ./data `
  --model wrn28x10 `
  --epochs 200 `
  --batch_size 128 `
  --opt sgd `
  --lr 0.1 `
  --momentum 0.9 `
  --wd 5e-4 `
  --sched cosine `
  --warmup 5 `
  --randaugment N=2,M=9 `
  --mixup 0.2 `
  --label_smoothing 0.1 `
  --ema 0.999 `
  --amp `
  --seed 42 `
  --out runs/wrn28x10_ra_mixup
```

### Evaluation

```powershell
python -m src.eval `
  --ckpt runs/wrn28x10_ra_mixup/best.ckpt `
  --data_root ./data `
  --dataset cifar10 `
  --save_confusion_matrix
```

## Expected Results

- **Target Accuracy**: ≥96% on CIFAR-10 test set
- **Training Time**: ~8-12 hours on RTX 3060 Laptop (200 epochs)
- **GPU Memory**: ~6-8 GB

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | `./data` | Path to dataset |
| `--model` | `wrn28x10` | Model architecture |
| `--epochs` | `200` | Number of training epochs |
| `--batch_size` | `128` | Batch size |
| `--lr` | `0.1` | Base learning rate |
| `--momentum` | `0.9` | SGD momentum |
| `--wd` | `5e-4` | Weight decay |
| `--warmup` | `5` | Warmup epochs |
| `--mixup` | `0.2` | Mixup alpha parameter |
| `--label_smoothing` | `0.1` | Label smoothing factor |
| `--ema` | `0.999` | EMA decay rate |
| `--amp` | Flag | Enable mixed precision training |
| `--seed` | `42` | Random seed |
| `--out` | `runs/...` | Output directory |

## Outputs

Training produces:
- `runs/{experiment}/best.ckpt` - Best model checkpoint
- `runs/{experiment}/last.ckpt` - Last epoch checkpoint
- `runs/{experiment}/logs/` - TensorBoard logs
- `runs/{experiment}/confusion_matrix.png` - Confusion matrix visualization

## Monitoring Training

View training progress with TensorBoard:

```powershell
tensorboard --logdir runs/wrn28x10_ra_mixup/logs
```

Then open http://localhost:6006 in your browser.

## CIFAR-10 Classes

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

## References

- WideResNet: [Zagoruyko & Komodakis, 2016](https://github.com/szagoruyko/wide-residual-networks)
- RandAugment: [Cubuk et al., 2019](https://arxiv.org/abs/1909.13719)
- Mixup: [Zhang et al., 2017](https://arxiv.org/abs/1710.09412)
- CutMix: [Yun et al., 2019](https://arxiv.org/abs/1905.04899)

## Hardware Specifications

- **GPU**: NVIDIA RTX 3060 Laptop
- **VRAM**: 6GB
- **Recommended Batch Size**: 128 (adjust if OOM occurs)

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch size: `--batch_size 64`
- Disable mixed precision: remove `--amp` flag

### Slow Training
- Reduce number of workers: `--num_workers 2`
- Enable mixed precision if not already: `--amp`

### Low Accuracy
- Increase training epochs: `--epochs 300`
- Adjust learning rate: `--lr 0.05`
- Try different augmentation parameters

## License

This project is for educational purposes.
