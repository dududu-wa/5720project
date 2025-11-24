"""
Generate comprehensive training comparison chart with all three experiments:
- Baseline: 200 epochs (RandAugment + Mixup)
- Improved: 300 epochs (CutMix)
- Failed: Configuration that didn't converge well
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 10


def load_tensorboard_logs(log_dir):
    """Load all metrics from TensorBoard logs"""
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    data = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        data[tag] = {
            'steps': [e.step for e in events],
            'values': [e.value for e in events]
        }
    
    return data


def load_all_logs(log_dir):
    """Load and merge all TensorBoard event files in chronological order"""
    files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
    if not files:
        return None
    
    # Sort by file modification time to get chronological order
    files.sort(key=lambda f: os.path.getmtime(os.path.join(log_dir, f)))
    
    # Merge all data
    merged_data = {}
    
    for file in files:
        log_path = os.path.join(log_dir, file)
        print(f"  Reading: {file}")
        ea = event_accumulator.EventAccumulator(log_path)
        ea.Reload()
        
        for tag in ea.Tags()['scalars']:
            if tag not in merged_data:
                merged_data[tag] = {'steps': [], 'values': []}
            
            events = ea.Scalars(tag)
            for e in events:
                merged_data[tag]['steps'].append(e.step)
                merged_data[tag]['values'].append(e.value)
    
    return merged_data


def smooth_curve(values, weight=0.9):
    """Exponential moving average smoothing"""
    smoothed = []
    last = values[0] if len(values) > 0 else 0
    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


# Load data from all three experiments
print("Loading training logs...")

experiments = {
    'Baseline (200 epochs)': {
        'log_dir': 'runs/wrn28x10_ra_mixup/logs',
        'color': 'green',
        'best_acc': 96.98,
        'config': 'RandAugment + Mixup'
    },
    'Improved (300 epochs)': {
        'log_dir': 'runs/wrn28x10_cutmix/logs',
        'color': 'blue',
        'best_acc': 97.51,
        'config': 'CutMix + Enhanced Aug'
    },
    'Failed Config': {
        'log_dir': 'runs/wrn28x10_improved/logs',
        'color': 'red',
        'best_acc': 80.95,
        'config': 'Suboptimal Settings',
        'style': '--'
    }
}

# Load all data
all_data = {}
for name, config in experiments.items():
    print(f"\nLoading {name}:")
    merged_data = load_all_logs(config['log_dir'])
    if merged_data:
        all_data[name] = merged_data
        all_data[name]['config'] = config
    else:
        print(f"Warning: No logs found for {name}")

# Create comprehensive comparison figure
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Main Validation Accuracy Comparison (Large, top-left)
ax1 = fig.add_subplot(gs[0:2, 0:2])
for name, data in all_data.items():
    config = data['config']
    if 'val/acc' in data:
        epochs = np.array(data['val/acc']['steps'])
        accs = np.array(data['val/acc']['values'])
        
        style = config.get('style', '-')
        ax1.plot(epochs, accs, 
                label=f"{name} (Best: {config['best_acc']:.2f}%)",
                color=config['color'],
                linewidth=2,
                linestyle=style,
                alpha=0.8)

ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Training Comparison: Validation Accuracy', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([40, 100])

# 2. Validation Loss Comparison
ax2 = fig.add_subplot(gs[0, 2])
for name, data in all_data.items():
    config = data['config']
    if 'val/loss' in data:
        epochs = np.array(data['val/loss']['steps'])
        losses = np.array(data['val/loss']['values'])
        
        style = config.get('style', '-')
        ax2.plot(epochs, losses,
                color=config['color'],
                linewidth=2,
                linestyle=style,
                alpha=0.8)

ax2.set_xlabel('Epoch', fontsize=10)
ax2.set_ylabel('Validation Loss', fontsize=10)
ax2.set_title('Validation Loss Curves', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend([name for name in all_data.keys()], fontsize=9)

# 3. Training Loss Comparison
ax3 = fig.add_subplot(gs[1, 2])
for name, data in all_data.items():
    config = data['config']
    if 'train/loss' in data:
        steps = np.array(data['train/loss']['steps'])
        losses = np.array(data['train/loss']['values'])
        # Smooth the training loss for better visualization
        smoothed = smooth_curve(losses, weight=0.95)
        
        style = config.get('style', '-')
        ax3.plot(steps, smoothed,
                color=config['color'],
                linewidth=1.5,
                linestyle=style,
                alpha=0.7)

ax3.set_xlabel('Training Step', fontsize=10)
ax3.set_ylabel('Training Loss', fontsize=10)
ax3.set_title('Training Loss Progress', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Training Accuracy Comparison
ax4 = fig.add_subplot(gs[2, 0])
for name, data in all_data.items():
    config = data['config']
    if 'train/acc' in data:
        steps = np.array(data['train/acc']['steps'])
        accs = np.array(data['train/acc']['values'])
        # Smooth training accuracy
        smoothed = smooth_curve(accs, weight=0.95)
        
        style = config.get('style', '-')
        ax4.plot(steps, smoothed,
                color=config['color'],
                linewidth=1.5,
                linestyle=style,
                alpha=0.7)

ax4.set_xlabel('Training Step', fontsize=10)
ax4.set_ylabel('Training Accuracy (%)', fontsize=10)
ax4.set_title('Training Accuracy Progress', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3)

# 5. EMA Validation Accuracy
ax5 = fig.add_subplot(gs[2, 1])
for name, data in all_data.items():
    config = data['config']
    if 'val/ema_acc' in data:
        epochs = np.array(data['val/ema_acc']['steps'])
        accs = np.array(data['val/ema_acc']['values'])
        
        style = config.get('style', '-')
        ax5.plot(epochs, accs,
                color=config['color'],
                linewidth=2,
                linestyle=style,
                alpha=0.8)

ax5.set_xlabel('Epoch', fontsize=10)
ax5.set_ylabel('EMA Val Accuracy (%)', fontsize=10)
ax5.set_title('EMA Model Performance', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. Learning Rate Schedule
ax6 = fig.add_subplot(gs[2, 2])
for name, data in all_data.items():
    config = data['config']
    if 'train/lr' in data:
        steps = np.array(data['train/lr']['steps'])
        lrs = np.array(data['train/lr']['values'])
        
        style = config.get('style', '-')
        ax6.plot(steps, lrs,
                color=config['color'],
                linewidth=2,
                linestyle=style,
                alpha=0.8)

ax6.set_xlabel('Training Step', fontsize=10)
ax6.set_ylabel('Learning Rate', fontsize=10)
ax6.set_title('Learning Rate Schedule', fontsize=11, fontweight='bold')
ax6.set_yscale('log')
ax6.grid(True, alpha=0.3)

# Add overall title
fig.suptitle('CIFAR-10 Training Comparison: Baseline vs Improved vs Failed Configuration', 
             fontsize=16, fontweight='bold', y=0.995)

# Add summary statistics box
summary_text = "Model Configuration Summary:\n\n"
for name, data in all_data.items():
    config = data['config']
    best_acc = config['best_acc']
    config_name = config['config']
    summary_text += f"{name}:\n"
    summary_text += f"  - Best Acc: {best_acc:.2f}%\n"
    summary_text += f"  - Config: {config_name}\n\n"

# Add text box with summary
props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
fig.text(0.02, 0.02, summary_text, fontsize=9, 
         verticalalignment='bottom', bbox=props, family='monospace')

plt.tight_layout()
plt.savefig('training_comparison_complete.png', dpi=300, bbox_inches='tight')
print("\n✓ Chart saved as 'training_comparison_complete.png'")

# Print detailed statistics
print("\n" + "="*70)
print("DETAILED TRAINING STATISTICS")
print("="*70)

for name, data in all_data.items():
    config = data['config']
    print(f"\n{name}:")
    print(f"  Configuration: {config['config']}")
    print(f"  Best Validation Accuracy: {config['best_acc']:.2f}%")
    
    if 'val/acc' in data:
        accs = data['val/acc']['values']
        epochs = data['val/acc']['steps']
        print(f"  Total Epochs: {max(epochs)}")
        print(f"  Final Accuracy: {accs[-1]:.2f}%")
        print(f"  Max Accuracy: {max(accs):.2f}%")
        
        # Find when reached 90%, 95%
        for threshold in [90, 95]:
            idx = next((i for i, v in enumerate(accs) if v >= threshold), None)
            if idx:
                print(f"  Reached {threshold}% at epoch: {epochs[idx]}")

print("\n" + "="*70)
