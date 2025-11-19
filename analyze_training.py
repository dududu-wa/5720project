"""
Extract and visualize training data from TensorBoard logs
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def extract_tensorboard_data(log_dir):
    """Extract scalar data from TensorBoard logs"""
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_files.append(os.path.join(root, file))
    
    if not event_files:
        print("⚠ No TensorBoard logs found!")
        return None
    
    # Use the latest event file
    latest_event = max(event_files, key=os.path.getmtime)
    print(f"Reading: {os.path.basename(latest_event)}")
    
    # Load event file
    ea = event_accumulator.EventAccumulator(latest_event)
    ea.Reload()
    
    # Get available tags
    tags = ea.Tags()
    scalar_tags = tags.get('scalars', [])
    
    data = {}
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {'steps': steps, 'values': values}
    
    return data


def create_training_table(data):
    """Create training metrics table"""
    if not data or 'val/acc' not in data:
        print("⚠ No validation accuracy data found")
        return None
    
    # Get validation data
    val_steps = data['val/acc']['steps']
    val_acc = data['val/acc']['values']
    val_loss = data['val/loss']['values'] if 'val/loss' in data else [0] * len(val_acc)
    
    # Get EMA validation data if available
    ema_val_acc = data.get('val/ema_acc', {}).get('values', [None] * len(val_acc))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Epoch': [step + 1 for step in val_steps],
        'Val Loss': [f'{loss:.4f}' for loss in val_loss],
        'Val Acc (%)': [f'{acc:.2f}' for acc in val_acc],
        'EMA Val Acc (%)': [f'{acc:.2f}' if acc is not None else 'N/A' for acc in ema_val_acc]
    })
    
    return df


def plot_training_curves(data, save_path='outputs/training_curves.png'):
    """Plot training and validation curves"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Progress', fontsize=20, fontweight='bold', y=0.995)
    
    # Plot 1: Training Loss
    if 'train/loss' in data:
        ax = axes[0, 0]
        steps = data['train/loss']['steps']
        values = data['train/loss']['values']
        ax.plot(steps, values, linewidth=1.5, color='#2E86AB', alpha=0.7)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Training Accuracy
    if 'train/acc' in data:
        ax = axes[0, 1]
        steps = data['train/acc']['steps']
        values = data['train/acc']['values']
        ax.plot(steps, values, linewidth=1.5, color='#A23B72', alpha=0.7)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Training Accuracy', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Validation Loss
    if 'val/loss' in data:
        ax = axes[1, 0]
        steps = data['val/loss']['steps']
        values = data['val/loss']['values']
        epochs = [s + 1 for s in steps]
        ax.plot(epochs, values, linewidth=2, color='#F18F01', marker='o', markersize=4)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Validation Loss', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Validation Accuracy (with EMA if available)
    if 'val/acc' in data:
        ax = axes[1, 1]
        steps = data['val/acc']['steps']
        values = data['val/acc']['values']
        epochs = [s + 1 for s in steps]
        ax.plot(epochs, values, linewidth=2, color='#6A994E', marker='o', 
                markersize=4, label='Val Acc')
        
        if 'val/ema_acc' in data:
            ema_values = data['val/ema_acc']['values']
            ax.plot(epochs, ema_values, linewidth=2, color='#BC4B51', marker='s', 
                    markersize=4, label='EMA Val Acc')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add best accuracy annotation
        best_acc = max(values)
        best_epoch = epochs[values.index(best_acc)]
        ax.axhline(y=best_acc, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.text(0.02, 0.98, f'Best: {best_acc:.2f}% @ Epoch {best_epoch}', 
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def create_summary_table(data):
    """Create summary statistics table"""
    if 'val/acc' in data:
        val_acc = data['val/acc']['values']
        val_loss = data['val/loss']['values'] if 'val/loss' in data else []
        
        summary = {
            'Metric': ['Best Val Acc', 'Latest Val Acc', 'Best Val Loss', 'Latest Val Loss', 'Epochs Completed'],
            'Value': [
                f"{max(val_acc):.2f}%" if val_acc else 'N/A',
                f"{val_acc[-1]:.2f}%" if val_acc else 'N/A',
                f"{min(val_loss):.4f}" if val_loss else 'N/A',
                f"{val_loss[-1]:.4f}" if val_loss else 'N/A',
                str(len(val_acc))
            ]
        }
        
        if 'val/ema_acc' in data:
            ema_acc = data['val/ema_acc']['values']
            summary['Metric'].insert(1, 'Best EMA Val Acc')
            summary['Value'].insert(1, f"{max(ema_acc):.2f}%")
        
        return pd.DataFrame(summary)
    return None


def create_epoch_table(data, last_n=20):
    """Create table for last N epochs"""
    if not data or 'val/acc' not in data:
        return None
    
    val_steps = data['val/acc']['steps']
    val_acc = data['val/acc']['values']
    val_loss = data['val/loss']['values'] if 'val/loss' in data else [0] * len(val_acc)
    ema_val_acc = data.get('val/ema_acc', {}).get('values', [None] * len(val_acc))
    
    # Get last N epochs
    start_idx = max(0, len(val_steps) - last_n)
    
    df = pd.DataFrame({
        'Epoch': [step + 1 for step in val_steps[start_idx:]],
        'Val Loss': [f'{loss:.4f}' for loss in val_loss[start_idx:]],
        'Val Acc': [f'{acc:.2f}%' for acc in val_acc[start_idx:]],
        'EMA Val Acc': [f'{acc:.2f}%' if acc is not None else 'N/A' for acc in ema_val_acc[start_idx:]]
    })
    
    # Add improvement indicator
    improvements = []
    best_so_far = 0
    for acc in val_acc[start_idx:]:
        if acc > best_so_far:
            improvements.append('↑ New Best')
            best_so_far = acc
        else:
            improvements.append('')
    df['Status'] = improvements
    
    return df


def main():
    os.makedirs('outputs', exist_ok=True)
    
    print("\n" + "="*70)
    print("Training Data Analysis")
    print("="*70 + "\n")
    
    log_dir = 'runs/wrn28x10_ra_mixup/logs'
    
    if not os.path.exists(log_dir):
        print(f"⚠ Log directory not found: {log_dir}")
        print("Please ensure training has started.")
        return
    
    # Extract data
    print("1. Extracting TensorBoard data...")
    data = extract_tensorboard_data(log_dir)
    
    if not data:
        return
    
    print(f"   Found {len(data)} metrics")
    
    # Create summary table
    print("\n2. Creating summary table...")
    summary_df = create_summary_table(data)
    if summary_df is not None:
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(summary_df.to_string(index=False))
        print("="*50)
        summary_df.to_csv('outputs/training_summary.csv', index=False)
        print("\n✓ Saved: outputs/training_summary.csv")
    
    # Create recent epochs table
    print("\n3. Creating recent epochs table...")
    epoch_df = create_epoch_table(data, last_n=20)
    if epoch_df is not None:
        print("\n" + "="*70)
        print("LAST 20 EPOCHS")
        print("="*70)
        print(epoch_df.to_string(index=False))
        print("="*70)
        epoch_df.to_csv('outputs/recent_epochs.csv', index=False)
        print("\n✓ Saved: outputs/recent_epochs.csv")
    
    # Create full training table
    print("\n4. Creating complete training table...")
    full_df = create_training_table(data)
    if full_df is not None:
        full_df.to_csv('outputs/full_training_log.csv', index=False)
        print(f"✓ Saved: outputs/full_training_log.csv ({len(full_df)} epochs)")
    
    # Plot training curves
    print("\n5. Generating training curves...")
    plot_training_curves(data)
    
    print("\n" + "="*70)
    print("✓ All tables and plots saved to 'outputs/' directory")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
