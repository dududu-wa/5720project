"""
Training script for CIFAR-10 with WRN-28-10, RandAugment, Mixup, and EMA
"""
import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models.wrn import wrn28x10, wrn40x4
from src.dataset import get_cifar10_dataloaders
from src.augment.mixup_cutmix import mixup_data, cutmix_data, mixup_criterion
from src.utils import (
    set_seed, AverageMeter, EMA, save_checkpoint, 
    accuracy, WarmupCosineSchedule
)


def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 Training')
    
    # Data
    parser.add_argument('--data_root', default='./data', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str)
    
    # Model
    parser.add_argument('--model', default='wrn28x10', type=str, 
                        choices=['wrn28x10', 'wrn40x4'])
    
    # Training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    
    # Optimizer
    parser.add_argument('--opt', default='sgd', type=str)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=5e-4, type=float)
    
    # Scheduler
    parser.add_argument('--sched', default='cosine', type=str)
    parser.add_argument('--warmup', default=5, type=int)
    
    # Augmentation
    parser.add_argument('--randaugment', default='N=2,M=9', type=str)
    parser.add_argument('--mixup', default=0.2, type=float)
    parser.add_argument('--cutmix', default=0.0, type=float)
    parser.add_argument('--label_smoothing', default=0.1, type=float)
    
    # EMA
    parser.add_argument('--ema', default=0.999, type=float)
    
    # Mixed precision
    parser.add_argument('--amp', action='store_true', default=True)
    
    # Misc
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--out', default='runs/wrn28x10_ra_mixup', type=str)
    parser.add_argument('--resume', default='', type=str)
    
    return parser.parse_args()


def train_epoch(model, train_loader, criterion, optimizer, scheduler, 
                ema, args, epoch, writer, scaler):
    """Train for one epoch"""
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.cuda(), targets.cuda()
        
        # Apply Mixup or CutMix
        use_mixup = args.mixup > 0 and torch.rand(1).item() > 0.5
        use_cutmix = args.cutmix > 0 and torch.rand(1).item() > 0.5
        
        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.mixup)
        elif use_cutmix:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, args.cutmix)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if args.amp:
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                if use_mixup or use_cutmix:
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            if use_mixup or use_cutmix:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
        
        # Update EMA
        if ema is not None:
            ema.update()
        
        # Measure accuracy and record loss
        acc1 = accuracy(outputs, targets)[0]
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{top1.avg:.2f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # Log to tensorboard
        global_step = epoch * len(train_loader) + batch_idx
        if writer is not None:
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/acc', acc1.item(), global_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
    
    return losses.avg, top1.avg


def validate(model, test_loader, criterion):
    """Validate the model"""
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Validating'):
            inputs, targets = inputs.cuda(), targets.cuda()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            acc1 = accuracy(outputs, targets)[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
    
    return losses.avg, top1.avg


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Fix for Windows multiprocessing
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Create tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.out, 'logs'))
    
    # Parse RandAugment parameters
    ra_params = dict(item.split('=') for item in args.randaugment.split(','))
    ra_n = int(ra_params.get('N', 2))
    ra_m = int(ra_params.get('M', 9))
    
    # Create dataloaders
    print(f"Loading CIFAR-10 dataset from {args.data_root}")
    train_loader, test_loader = get_cifar10_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_randaugment=True,
        ra_n=ra_n,
        ra_m=ra_m
    )
    
    # Create model
    print(f"Creating model: {args.model}")
    if args.model == 'wrn28x10':
        model = wrn28x10(num_classes=10)
    elif args.model == 'wrn40x4':
        model = wrn40x4(num_classes=10)
    model = model.cuda()
    
    # Create optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
        nesterov=True
    )
    
    # Create scheduler
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_epochs=args.warmup,
        total_epochs=args.epochs,
        base_lr=args.lr
    )
    
    # Create loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda()
    
    # Create EMA
    ema = EMA(model, decay=args.ema) if args.ema > 0 else None
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if args.amp else None
    
    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0.0
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(start_epoch, args.epochs):
        # Update learning rate
        lr = scheduler.step(epoch)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            ema, args, epoch, writer, scaler
        )
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion)
        
        # Validate with EMA
        if ema is not None:
            ema.apply_shadow()
            ema_val_loss, ema_val_acc = validate(model, test_loader, criterion)
            ema.restore()
            print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, '
                  f'Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, '
                  f'EMA Val Acc={ema_val_acc:.2f}%')
            
            # Use EMA accuracy for best model
            current_acc = ema_val_acc
        else:
            print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, '
                  f'Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%')
            current_acc = val_acc
        
        # Log to tensorboard
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)
        if ema is not None:
            writer.add_scalar('val/ema_acc', ema_val_acc, epoch)
        
        # Save checkpoint
        is_best = current_acc > best_acc
        if is_best:
            best_acc = current_acc
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'args': args
        }
        
        if ema is not None:
            checkpoint['ema_shadow'] = ema.shadow
        
        # Save last checkpoint
        save_checkpoint(checkpoint, os.path.join(args.out, 'last.ckpt'))
        
        # Save best checkpoint
        if is_best:
            save_checkpoint(checkpoint, os.path.join(args.out, 'best.ckpt'))
            print(f'New best accuracy: {best_acc:.2f}%')
    
    writer.close()
    print(f"\nTraining completed! Best accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
