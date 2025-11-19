# Ultra High Performance Training Script
# Using WRN-40-4 for better capacity

$EXP = "wrn40x4_ultra"

Write-Host "=== Ultra Performance Training ===" -ForegroundColor Cyan
Write-Host "Model: WRN-40-4 (deeper network)" -ForegroundColor Yellow
Write-Host ""

conda activate pythonProject1

Write-Host "Starting ultra training..." -ForegroundColor Green
python -m src.train `
  --dataset cifar10 `
  --data_root ./data `
  --model wrn40x4 `
  --epochs 300 `
  --batch_size 128 `
  --num_workers 4 `
  --opt sgd `
  --lr 0.1 `
  --momentum 0.9 `
  --wd 5e-4 `
  --sched cosine `
  --warmup 10 `
  --randaugment N=2,M=10 `
  --mixup 0.5 `
  --cutmix 0.5 `
  --label_smoothing 0.1 `
  --ema 0.9995 `
  --amp `
  --seed 42 `
  --out runs/$EXP

Write-Host "`nEvaluating model..." -ForegroundColor Green
python -m src.eval `
  --ckpt runs/$EXP/best.ckpt `
  --data_root ./data `
  --dataset cifar10 `
  --save_confusion_matrix

Write-Host "`nDone!" -ForegroundColor Green
