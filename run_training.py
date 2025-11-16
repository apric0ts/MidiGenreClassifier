"""
Script to run the enhanced training pipeline.
"""
import os
from collections import Counter

import genreclassifier as gc
from genreclassifier.train_enhanced import train_model
# from genreclassifier.training2 import train_model


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
        files_walked_count=None  # None means get all info
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
        train_split=0.7,          # 70% for training
        val_split=0.15,           # 15% for validation
        test_split=0.15,          # 15% for testing
        batch_size=32,            # Adjust based on your GPU memory
        num_epochs=200,           # Maximum epochs (early stopping will likely stop earlier)
        learning_rate=5e-4,       # Lower learning rate for better convergence
        hidden_dims=[512, 384, 256, 128],  # Balanced network architecture
        dropout=0.3,             # Balanced dropout for regularization
        early_stopping_patience=20,  # More patience for better convergence
        combine_rare=True,        # Combine rare genres into 'Other'
        min_samples_per_genre=100  # Minimum samples to keep a genre separate
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
    
    # Optionally save the model (skip prompt in non-interactive mode)
    try:
        save_model = input("\nSave model? (y/n): ").lower().strip()
    except (EOFError, KeyboardInterrupt):
        save_model = 'n'
        print("\n(Skipping model save in non-interactive mode)")
    
    if save_model == 'y':
        import torch
        model_path = "trained_genre_classifier.pth"
        torch.save({
            'model_state_dict': results['model'].state_dict(),
            'genre_to_idx': results['genre_to_idx'],
            'idx_to_genre': results['idx_to_genre'],
            'scaler': results['train_dataset'].scaler,
            'test_metrics': test_metrics
        }, model_path)
        print(f"Model saved to {model_path}")
    
    print("\nDone!")

