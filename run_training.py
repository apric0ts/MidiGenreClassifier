"""
Script to run the enhanced training pipeline.
"""
import os
from collections import Counter
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

import genreclassifier as gc
from genreclassifier.train_enhanced import train_model
from visualizations import generate_all_visualizations

# Create visualizations folder
os.makedirs('visualizations/explainability', exist_ok=True)

def run_standalone_explainability(results, test_loader, device):
    """
    Run explainability analysis on trained model results
    Call this function after your training is complete
    """
    print("\n" + "=" * 60)
    print("STANDALONE EXPLAINABILITY ANALYSIS")
    print("=" * 60)
    
    model = results['model']
    genre_names = list(results['idx_to_genre'].values())
    
    # 1. Feature Importance Analysis
    print("1. Analyzing feature importance...")
    
    feature_importances = []
    
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        X_batch.requires_grad = True
        
        outputs = model(X_batch)
        
        # Use gradients w.r.t. input features
        model.zero_grad()
        outputs.sum().backward()
        
        # Average absolute gradients across batch
        gradients = X_batch.grad.abs().mean(dim=0).cpu().numpy()
        feature_importances.append(gradients)
    
    # Average across all batches
    avg_importances = np.mean(feature_importances, axis=0)
    
    # Plot top features
    top_n = min(15, len(avg_importances))
    sorted_idx = avg_importances.argsort()[::-1][:top_n]
    
    # Create feature names
    num_features = avg_importances.shape[0]
    feature_names = [f"Feature_{i}" for i in range(num_features)]
    
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(top_n)
    
    top_features = [feature_names[i] for i in sorted_idx]
    top_scores = avg_importances[sorted_idx]
    
    bars = plt.barh(y_pos, top_scores, color='lightblue', alpha=0.7, edgecolor='navy')
    
    plt.xlabel('Feature Importance (Gradient Magnitude)', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Most Important MIDI Features', fontsize=14, fontweight='bold')
    plt.yticks(y_pos, top_features)
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, score in zip(bars, top_scores):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{score:.4f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('visualizations/explainability/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: visualizations/explainability/feature_importance.png")
    
    # 2. Confidence Analysis
    print("2. Analyzing prediction confidence...")
    
    all_confidences = []
    all_correct = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, 1)
            
            # Check if predictions are correct
            correct = (predictions.cpu() == y_batch).numpy()
            
            all_confidences.extend(confidences.cpu().numpy())
            all_correct.extend(correct)
    
    all_confidences = np.array(all_confidences)
    all_correct = np.array(all_correct)
    
    # Plot confidence distributions
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(all_confidences, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Prediction Confidence', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.title('Distribution of Prediction Confidences', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    if len(all_correct) > 0:
        correct_confidences = all_confidences[all_correct] if np.any(all_correct) else np.array([])
        incorrect_confidences = all_confidences[~all_correct] if np.any(~all_correct) else np.array([])
        
        if len(correct_confidences) > 0:
            plt.hist(correct_confidences, bins=15, alpha=0.7, label='Correct', color='green')
        if len(incorrect_confidences) > 0:
            plt.hist(incorrect_confidences, bins=15, alpha=0.7, label='Incorrect', color='red')
        
        plt.xlabel('Prediction Confidence', fontweight='bold')
        plt.ylabel('Frequency', fontweight='bold')
        plt.title('Confidence: Correct vs Incorrect', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/explainability/confidence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: visualizations/explainability/confidence_analysis.png")
    
    # 3. Misclassification Analysis
    print("3. Analyzing misclassifications...")
    
    confusion_pairs = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predictions = torch.max(outputs, 1)
            
            for true_idx, pred_idx in zip(y_batch.numpy(), predictions.cpu().numpy()):
                if true_idx != pred_idx:
                    confusion_pairs.append((true_idx, pred_idx))
    
    # Count confusion frequencies
    confusion_counts = {}
    for true_idx, pred_idx in confusion_pairs:
        pair = (true_idx, pred_idx)
        confusion_counts[pair] = confusion_counts.get(pair, 0) + 1
    
    # Get top confusions
    top_confusions = sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Plot
    if top_confusions:
        plt.figure(figsize=(12, 6))
        
        confusion_labels = []
        counts = []
        
        for (true_idx, pred_idx), count in top_confusions:
            true_genre = genre_names[true_idx]
            pred_genre = genre_names[pred_idx]
            confusion_labels.append(f"{true_genre} → {pred_genre}")
            counts.append(count)
        
        y_pos = np.arange(len(confusion_labels))
        
        bars = plt.barh(y_pos, counts, color='salmon', edgecolor='darkred', alpha=0.7)
        plt.xlabel('Number of Misclassifications', fontweight='bold')
        plt.title('Top Genre Confusions', fontweight='bold')
        plt.yticks(y_pos, confusion_labels)
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{count}', ha='left', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('visualizations/explainability/misclassification_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Saved: visualizations/explainability/misclassification_analysis.png")
    else:
        print("✓ No misclassifications found in test set!")
    
    # 4. Genre-wise Performance Analysis
    print("4. Analyzing genre-wise performance...")
    
    genre_correct = {genre: 0 for genre in genre_names}
    genre_total = {genre: 0 for genre in genre_names}
    genre_confidence = {genre: [] for genre in genre_names}
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, 1)
            
            for true_idx, pred_idx, conf in zip(y_batch.numpy(), predictions.cpu().numpy(), confidences.cpu().numpy()):
                true_genre = genre_names[true_idx]
                genre_total[true_genre] += 1
                genre_confidence[true_genre].append(conf)
                if true_idx == pred_idx:
                    genre_correct[true_genre] += 1
    
    # Calculate accuracy per genre
    genre_accuracy = {}
    for genre in genre_names:
        if genre_total[genre] > 0:
            genre_accuracy[genre] = genre_correct[genre] / genre_total[genre]
        else:
            genre_accuracy[genre] = 0
    
    # Plot genre performance
    plt.figure(figsize=(10, 6))
    genres_sorted = sorted(genre_accuracy.items(), key=lambda x: x[1])
    genres_list = [g[0] for g in genres_sorted]
    accuracies = [g[1] for g in genres_sorted]
    
    colors = ['red' if acc < 0.3 else 'orange' if acc < 0.5 else 'green' for acc in accuracies]
    
    bars = plt.barh(genres_list, accuracies, color=colors, alpha=0.7)
    plt.xlabel('Accuracy', fontweight='bold')
    plt.title('Genre-wise Classification Accuracy', fontweight='bold')
    plt.xlim(0, 1.0)
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{acc:.2f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/explainability/genre_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: visualizations/explainability/genre_accuracy.png")
    
    # Print statistics
    print("\n" + "=" * 50)
    print("EXPLAINABILITY SUMMARY")
    print("=" * 50)
    print(f"Total test samples: {len(all_confidences)}")
    print(f"Average confidence: {np.mean(all_confidences):.3f}")
    
    if len(all_correct) > 0 and np.any(all_correct):
        correct_confidences = all_confidences[all_correct]
        print(f"Correct predictions confidence: {np.mean(correct_confidences):.3f}")
    
    if len(all_correct) > 0 and np.any(~all_correct):
        incorrect_confidences = all_confidences[~all_correct]
        print(f"Incorrect predictions confidence: {np.mean(incorrect_confidences):.3f}")
    
    print(f"Number of misclassifications: {len(confusion_pairs)}")
    print(f"Top 3 influential features: {top_features[:3]}")
    
    print("\nGenre Performance Summary:")
    for genre, acc in genres_sorted:
        print(f"  {genre}: {acc:.2f} accuracy")
    
    print("=" * 50)


if __name__ == "__main__":
    """
    Project structure:
    outer-folder
    - MidiGenreClassifier (our repo)
    - MidiFiles (self explanatory)
    - matches_scores.json (matching midi files to tracks in MSD)
    - msd_tagtraum_cd1.cls (genres)
    """
    
    # Project root directory (parent of MidiGenreClassifier)
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    midi_files_path = os.path.join(parent_dir, "MidiFiles")
    match_scores_path = os.path.join(parent_dir, "match_scores.json")
    genres_path = os.path.join(parent_dir, "msd_tagtraum_cd1.cls")
    
    print("=" * 60)
    print("Loading MIDI tracks and features...")
    print("=" * 60)
    
    # Load tracks (will use cache if available)
    tracks: list[gc.Track] = gc.get_all_track_information(
        midi_files_path, 
        match_scores_path, 
        genres_path, 
        cache_path="midi_features_cache_2.pkl",
        files_walked_count=None  
    )
    
    print(f"\nLoaded {len(tracks)} tracks")
    
    # Display genre distribution
    genres = Counter([track.genre for track in tracks])
    print(f"\nGenre distribution:")
    for genre, count in genres.most_common():
        print(f"  {genre}: {count}")
    print(f"\nTotal unique genres: {len(genres)}")
    
    # Train the model
    print("\n" + "=" * 60)
    print("Training model with enhanced pipeline...")
    print("=" * 60)
    
    results = train_model(
        tracks=tracks,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        batch_size=32,
        num_epochs=200,
        learning_rate=5e-4,
        hidden_dims=[256, 128, 64],
        dropout=0.3,
        early_stopping_patience=20,
        combine_rare=True,
        min_samples_per_genre=100,
        use_class_weights=True,
        device=None
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    test_metrics = results['test_metrics']
    
    print(f"\nFinal Test Set Performance:")
    print(f"  Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1-Score: {test_metrics['f1_score']:.4f}")
    
    # Per-class performance report
    print("\n" + "=" * 60)
    print("Per-Class Performance:")
    print("=" * 60)
    
    # Get predictions
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
    
    print(classification_report(
        all_labels,
        all_preds,
        target_names=list(results['idx_to_genre'].values()),
        zero_division=0
    ))
    
    # Ask about visualizations
    try:
        generate_viz = input("\nGenerate visualizations? (y/n): ").lower().strip()
    except (EOFError, KeyboardInterrupt):
        generate_viz = 'n'
        print("\n(Skipping visualizations)")
    
    if generate_viz == 'y':
        generate_all_visualizations(results, all_labels, all_preds, device)
    
    # Save model?
    try:
        save_model = input("\nSave model? (y/n): ").lower().strip()
    except (EOFError, KeyboardInterrupt):
        save_model = 'n'
        print("\n(Skipping model save)")
    
    if save_model == 'y':
        model_path = "trained_genre_classifier.pth"
        torch.save({
            'model_state_dict': results['model'].state_dict(),
            'genre_to_idx': results['genre_to_idx'],
            'idx_to_genre': results['idx_to_genre'],
            'scaler': results['train_dataset'].scaler,
            'test_metrics': test_metrics,
            'history': results['history'] 
        }, model_path)
        print(f"Model saved to {model_path}")
    
    # Ask about explainability analysis
    try:
        run_explain = input("\nRun explainability analysis? (y/n): ").lower().strip()
    except (EOFError, KeyboardInterrupt):
        run_explain = 'n'
        print("\n(Skipping explainability analysis)")
    
    if run_explain == 'y':
        run_standalone_explainability(results, test_loader, device)
    
    print("\nDone!")
