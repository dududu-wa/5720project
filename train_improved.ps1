# Improved Training Script (PowerShell)
# Enhanced configuration for higher accuracy

$EXP = "wrn28x10_improved"

Write-Host "=== Improved Training Configuration ===" -ForegroundColor Cyan
Write-Host "Target: >97.5% accuracy" -ForegroundColor Yellow
Write-Host ""
Write-Host "Improvements:" -ForegroundColor Green
Write-Host "  1. Extended training to 300 epochs" -ForegroundColor White
Write-Host "  2. Stronger data augmentation (CutMix)" -ForegroundColor White
Write-Host "  3. Larger batch size (256 with gradient accumulation)" -ForegroundColor White
Write-Host "  4. Lower initial learning rate with longer warmup" -ForegroundColor White
Write-Host "  5. Higher weight decay for better regularization" -ForegroundColor White
Write-Host ""

conda activate pythonProject1

Write-Host "Starting improved training..." -ForegroundColor Green
python -m src.train `
  --dataset cifar10 `
  --data_root ./data `
  --model wrn28x10 `
  --epochs 300 `
  --batch_size 256 `
  --num_workers 4 `
  --opt sgd `
  --lr 0.2 `
  --momentum 0.9 `
  --wd 1e-3 `
  --sched cosine `
  --warmup 10 `
  --randaugment N=3,M=12 `
  --mixup 0.0 `
  --cutmix 1.0 `
  --label_smoothing 0.1 `
  --ema 0.9995 `
  --amp `
  --seed 42 `
  --out runs/$EXP

Write-Host "`nEvaluating improved model..." -ForegroundColor Green
python -m src.eval `
  --ckpt runs/$EXP/best.ckpt `
  --data_root ./data `
  --dataset cifar10 `
  --save_confusion_matrix

Write-Host "`nDone!" -ForegroundColor Green
