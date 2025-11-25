"""
Visualization for MIDI Genre Classifier.

AI used to help debug and generate graphs
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import (
    precision_recall_fscore_support, 
    confusion_matrix, 
    roc_curve, 
    auc
)
from sklearn.preprocessing import label_binarize
from itertools import cycle

import os

# Create visualizations folder
os.makedirs('visualizations', exist_ok=True)

def plot_training_history(history, save_path='visualizations/training_history.png'):
    """Plot training and validation metrics over epochs"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def plot_confusion_matrix(test_labels, test_preds, genre_names, save_path='visualizations/confusion_matrix.png'):
    """Beautiful confusion matrix heatmap"""
    cm = confusion_matrix(test_labels, test_preds)
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=genre_names, yticklabels=genre_names,
                cbar_kws={'label': 'Proportion'}, ax=ax, linewidths=0.5)
    
    ax.set_xlabel('Predicted Genre', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Genre', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def plot_per_class_metrics(test_labels, test_preds, genre_names, save_path='visualizations/per_class_metrics.png'):
    """Bar chart showing precision, recall, F1 for each genre"""
    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels, test_preds, zero_division=0
    )
    
    x = np.arange(len(genre_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
    
    ax.set_xlabel('Genre', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Genre Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(genre_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.0])
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def plot_multiclass_roc(model, test_loader, num_classes, genre_names, device, save_path='visualizations/roc_curves.png'):
    """Plot ROC curve for each class"""
    model.eval()
    
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # binary
    y_bin = label_binarize(all_labels, classes=range(num_classes))
    
    # Compute ROC curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = cycle(['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'])
    
    for i, color in zip(range(num_classes), colors):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{genre_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.50)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves for Each Genre', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def generate_all_visualizations(results, all_labels, all_preds, device):
    """Generate all visualizations at once"""
    print("\n" + "=" * 60)
    print("Generating Visualizations...")
    print("=" * 60)
    
    genre_names = list(results['idx_to_genre'].values())
    
    # Plot 1: Training history
    plot_training_history(results['history'])
    
    # plot 2: Confusion matrix
    plot_confusion_matrix(all_labels, all_preds, genre_names)
    
    # plot 3: Per-class metrics
    plot_per_class_metrics(all_labels, all_preds, genre_names)
    
    # Plot 4: ROC curves
    test_loader = torch.utils.data.DataLoader(
        results['test_dataset'], 
        batch_size=32, 
        shuffle=False
    )
    plot_multiclass_roc(
        results['model'], 
        test_loader, 
        results['test_dataset'].num_classes,
        genre_names,
        device
    )
    
    print("\n" + "=" * 60)
    print("All visualizations saved!")
    print("=" * 60)