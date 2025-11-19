"""
Quick hyperparameter experiments for MIDI Genre Classifier.
Tests key configurations to detect overfitting and optimal learning rate.
"""
import os
from collections import Counter
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

import genreclassifier as gc
from genreclassifier.train_enhanced import train_model


def run_experiment(tracks, experiment_name, learning_rate, num_epochs, batch_size=32):
    """Run a single training experiment and save results."""
    print("\n" + "=" * 60)
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Learning Rate: {learning_rate}, Epochs: {num_epochs}")
    print("=" * 60)
    
    results = train_model(
        tracks=tracks,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        hidden_dims=[256, 128, 64],
        dropout=0.3,
        early_stopping_patience=20,
        combine_rare=True,
        min_samples_per_genre=350,
        use_class_weights=True,
        device=None
    )
    
    # Get test predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_loader = DataLoader(results['test_dataset'], batch_size=32, shuffle=False)
    model = results['model']
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    # Print results
    test_metrics = results['test_metrics']
    print(f"\nResults:")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"  Test F1-Score: {test_metrics['f1_score']:.4f}")
    
    return {
        'name': experiment_name,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'history': results['history'],
        'test_accuracy': test_metrics['accuracy'],
        'test_f1': test_metrics['f1_score'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall']
    }


def plot_all_experiments(experiments, save_dir='visualizations/experiments'):
    """Create comparison plots for all experiments."""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Training curves comparison (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    for i, exp in enumerate(experiments):
        history = exp['history']
        label = exp['name']
        color = colors[i % len(colors)]
        
        # Train loss
        axes[0, 0].plot(history['train_loss'], label=label, linewidth=2, alpha=0.8, color=color)
        # Val loss
        axes[0, 1].plot(history['val_loss'], label=label, linewidth=2, alpha=0.8, color=color)
        # Train acc
        axes[1, 0].plot(history['train_acc'], label=label, linewidth=2, alpha=0.8, color=color)
        # Val acc
        axes[1, 1].plot(history['val_acc'], label=label, linewidth=2, alpha=0.8, color=color)
    
    axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Loss', fontsize=11)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Training Accuracy', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('Accuracy (%)', fontsize=11)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('Accuracy (%)', fontsize=11)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Training Curves Comparison', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_dir}/training_curves_comparison.png")
    
    # 2. Overfitting analysis (train-val gap)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for i, exp in enumerate(experiments):
        history = exp['history']
        label = exp['name']
        color = colors[i % len(colors)]
        
        # Loss gap
        loss_gap = np.array(history['val_loss']) - np.array(history['train_loss'])
        axes[0].plot(loss_gap, label=label, linewidth=2, alpha=0.8, color=color)
        
        # Accuracy gap
        acc_gap = np.array(history['train_acc']) - np.array(history['val_acc'])
        axes[1].plot(acc_gap, label=label, linewidth=2, alpha=0.8, color=color)
    
    axes[0].set_title('Loss Gap (Val - Train)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss Gap', fontsize=11)
    axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1.5)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].text(0.02, 0.98, '↑ Higher = More Overfitting', 
                transform=axes[0].transAxes, fontsize=10, 
                verticalalignment='top', style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    axes[1].set_title('Accuracy Gap (Train - Val)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Accuracy Gap (%)', fontsize=11)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1.5)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].text(0.02, 0.98, '↑ Higher = More Overfitting', 
                transform=axes[1].transAxes, fontsize=10, 
                verticalalignment='top', style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/overfitting_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_dir}/overfitting_analysis.png")
    
    # 3. Final test performance comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = [exp['name'] for exp in experiments]
    test_acc = [exp['test_accuracy'] for exp in experiments]
    test_f1 = [exp['test_f1'] * 100 for exp in experiments]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, test_acc, width, label='Test Accuracy', color='#3498db')
    bars2 = ax.bar(x + width/2, test_f1, width, label='Test F1-Score', color='#e74c3c')
    
    ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Test Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 60])
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/test_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_dir}/test_performance_comparison.png")
    
    # 4. Summary table
    print("\n" + "=" * 85)
    print("EXPERIMENT SUMMARY")
    print("=" * 85)
    print(f"{'Experiment':<30} {'LR':<10} {'Epochs':<10} {'Test Acc':<12} {'Test F1':<12}")
    print("-" * 85)
    for exp in experiments:
        print(f"{exp['name']:<30} {exp['learning_rate']:<10} {exp['num_epochs']:<10} "
              f"{exp['test_accuracy']:<12.2f} {exp['test_f1']:<12.4f}")
    print("=" * 85)


if __name__ == "__main__":
    # Load data
    print("=" * 60)
    print("Loading MIDI tracks and features...")
    print("=" * 60)
    
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    midi_files_path = os.path.join(parent_dir, "MidiFiles")
    match_scores_path = os.path.join(parent_dir, "match_scores.json")
    genres_path = os.path.join(parent_dir, "msd_tagtraum_cd1.cls")
    
    tracks = gc.get_all_track_information(
        midi_files_path, 
        match_scores_path, 
        genres_path, 
        cache_path="midi_features_cache_2.pkl",
        files_walked_count=None
    )
    
    print(f"\nLoaded {len(tracks)} tracks")
    
    # Run 3 quick experiments
    experiments = []
    
    # Experiment 1: Baseline (fewer epochs for speed)
    print("\n Running Experiment 1/3...")
    experiments.append(run_experiment(
        tracks, 
        "Baseline (LR=5e-4)", 
        learning_rate=5e-4, 
        num_epochs=100
    ))
    
    # Experiment 2: Test overfitting with more epochs
    print("\n Running Experiment 2/3...")
    experiments.append(run_experiment(
        tracks, 
        "More Epochs (LR=5e-4)", 
        learning_rate=5e-4, 
        num_epochs=200
    ))
    
    # Experiment 3: Different learning rate
    print("\n Running Experiment 3/3...")
    experiments.append(run_experiment(
        tracks, 
        "Higher LR (LR=1e-3)", 
        learning_rate=1e-3, 
        num_epochs=100
    ))
    
    # Generate comparison visualizations
    print("\n" + "=" * 60)
    print("Generating comparison visualizations...")
    print("=" * 60)
    plot_all_experiments(experiments)
    
    print("\nAll experiments complete!")