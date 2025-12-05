import os

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import torch

import genreclassifier as gc

MODEL_PATH = "trained_genre_classifier.pth"

def predict_genre(midi_path):
    device = "cpu"
    # 0. Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    genre_to_idx = checkpoint['genre_to_idx']
    idx_to_genre = checkpoint['idx_to_genre']
    scaler = checkpoint['scaler']

    # 1. Load model
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
    midi = gc.extract_data_from_midi(midi_path)
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
    return predicted_genre

class DragDropUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MIDI Genre Classifier")
        self.root.geometry("500x300")
        self.root.configure(bg="#1e1e1e")

        ttk.Label(root, text="Drag and Drop a MIDI File",
                  font=("Helvetica", 16)).pack(pady=20)
        self.drop_frame = tk.Frame(root, width=400, height=120, bg="#333333")
        self.drop_frame.pack(pady=10)
        self.drop_frame.pack_propagate(False)
        self.drop_label = ttk.Label(self.drop_frame, text="Drop .mid file here")
        self.drop_label.pack(expand=True)
        self.drop_frame.bind("<Button-1>", self.open_file_dialog)

        self.result_label = ttk.Label(root, text="", font=("Helvetica", 14))
        self.result_label.pack(pady=10)

        self.history_label = ttk.Label(root, text="", font=("Helvetica", 12))
        self.history_label.pack(pady=10)
        self.history = []

    def open_file_dialog(self, event=None):
        """Fallback file picker when clicking the box."""
        midi_path = filedialog.askopenfilename(
            title="Select MIDI file",
            filetypes=[("MIDI files", "*.mid *.midi")]
        )
        if midi_path:
            self.run_prediction(midi_path)

    def run_prediction(self, midi_path):
        try:
            genre = predict_genre(midi_path)

            filename = os.path.basename(midi_path)

            self.result_label.config(text=f"Predicted Genre: {genre}")

            self.history.append(f"{filename}: {genre}")

            history_text = "\n".join(self.history)
            self.history_label.config(text=history_text)

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    DragDropUI(root)
    root.mainloop()
