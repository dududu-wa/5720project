# Quick Test Training Script (PowerShell)
# Short test run with just 5 epochs to verify everything works

$EXP = "test_run"

Write-Host "Activating conda environment: pythonProject1" -ForegroundColor Green
conda activate pythonProject1

Write-Host "`nStarting quick test (5 epochs)..." -ForegroundColor Green
python -m src.train `
  --dataset cifar10 `
  --data_root ./data `
  --model wrn28x10 `
  --epochs 5 `
  --batch_size 128 `
  --opt sgd `
  --lr 0.1 `
  --momentum 0.9 `
  --wd 5e-4 `
  --sched cosine `
  --warmup 1 `
  --randaugment N=2,M=9 `
  --mixup 0.2 `
  --label_smoothing 0.1 `
  --ema 0.999 `
  --amp `
  --seed 42 `
  --out runs/$EXP

Write-Host "`nTest completed!" -ForegroundColor Green
