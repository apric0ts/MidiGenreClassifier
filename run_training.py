"""
Script to run the enhanced training pipeline.
"""
import os
from collections import Counter
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

import genreclassifier as gc
from genreclassifier.train_enhanced import train_model
from visualizations import generate_all_visualizations


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
    
    #save model?
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
    
    print("\nDone!")