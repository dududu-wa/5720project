# 监控训练进度的脚本

Write-Host "=== CIFAR-10 训练监控工具 ===" -ForegroundColor Cyan
Write-Host ""

# 检查训练日志
if (Test-Path "runs/wrn28x10_ra_mixup") {
    Write-Host "训练输出目录存在" -ForegroundColor Green
    
    # 显示最新的checkpoint
    if (Test-Path "runs/wrn28x10_ra_mixup/best.ckpt") {
        $bestCkpt = Get-Item "runs/wrn28x10_ra_mixup/best.ckpt"
        Write-Host "最佳模型: $($bestCkpt.Name) (最后修改: $($bestCkpt.LastWriteTime))" -ForegroundColor Green
    } else {
        Write-Host "尚未生成最佳模型" -ForegroundColor Yellow
    }
    
    if (Test-Path "runs/wrn28x10_ra_mixup/last.ckpt") {
        $lastCkpt = Get-Item "runs/wrn28x10_ra_mixup/last.ckpt"
        Write-Host "最新模型: $($lastCkpt.Name) (最后修改: $($lastCkpt.LastWriteTime))" -ForegroundColor Green
    } else {
        Write-Host "尚未生成checkpoint" -ForegroundColor Yellow
    }
    
    # 显示TensorBoard日志
    if (Test-Path "runs/wrn28x10_ra_mixup/logs") {
        Write-Host "`nTensorBoard日志目录存在" -ForegroundColor Green
        Write-Host "启动TensorBoard查看训练曲线:" -ForegroundColor Cyan
        Write-Host "  tensorboard --logdir runs/wrn28x10_ra_mixup/logs" -ForegroundColor White
        Write-Host "  然后在浏览器中打开: http://localhost:6006" -ForegroundColor White
    }
} else {
    Write-Host "训练尚未开始或输出目录不存在" -ForegroundColor Yellow
}

Write-Host "`n=== 使用说明 ===" -ForegroundColor Cyan
Write-Host "1. 查看训练日志: 检查上面的输出目录" -ForegroundColor White
Write-Host "2. 启动TensorBoard: tensorboard --logdir runs/wrn28x10_ra_mixup/logs" -ForegroundColor White
Write-Host "3. 如果训练中断,可以使用 --resume runs/wrn28x10_ra_mixup/last.ckpt 继续训练" -ForegroundColor White
Write-Host ""
