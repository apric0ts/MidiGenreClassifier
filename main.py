
import os
from collections import Counter

import numpy as np
import torch

import genreclassifier as gc

MODEL_PATH = "trained_genre_classifier.pth"
MIDI_PATH = "..." # insert here

if __name__ == "__main__":
    device = "cpu"
    # 0. Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    genre_to_idx = checkpoint['genre_to_idx']
    idx_to_genre = checkpoint['idx_to_genre']
    scaler = checkpoint['scaler']

    # Load model
    model = gc.EnhancedTrackToGenreMLP(
        input_dim=192,
        num_classes=len(idx_to_genre),
        hidden_dims=[256, 128, 64],
        dropout=0.35,
        use_batch_norm=True
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 2. Extract features from the MIDI file    
    midi = gc.extract_data_from_midi(MIDI_PATH)
    features: gc.MidiFeatures = midi.extract_feature_vector()
    x = features.astype(np.float32) # (192,) vector

    mask = gc.features.NORMALIZE_MASK
    mask = gc.NORMALIZE_MASK
    x_scaled = x.copy()

    if scaler is not None:
        x_scaled[mask] = scaler.transform(x[mask].reshape(1, -1))[0]

    x = torch.tensor(x_scaled).unsqueeze(0).to(device)

    # 3. Predict!
    with torch.no_grad():
        logits = model(x)
        pred_idx = torch.argmax(torch.softmax(logits, dim=1)).item()
    # 4. Map index to genre name
    predicted_genre = idx_to_genre[pred_idx]


    print("Predicted genre:", predicted_genre)