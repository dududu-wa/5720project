# CIFAR-10 Training Script (PowerShell)
# Activate conda environment and run training

$EXP = "wrn28x10_ra_mixup"

Write-Host "Activating conda environment: pythonProject1" -ForegroundColor Green
conda activate pythonProject1

Write-Host "`nStarting training..." -ForegroundColor Green
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
  --out runs/$EXP

Write-Host "`nStarting evaluation..." -ForegroundColor Green
python -m src.eval `
  --ckpt runs/$EXP/best.ckpt `
  --data_root ./data `
  --dataset cifar10 `
  --save_confusion_matrix

Write-Host "`nDone!" -ForegroundColor Green
