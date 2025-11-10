import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

from .features import Track, extract_feature_vector


class TrackDataset(Dataset):
    """
    PyTorch dataset for `Track` objects
    """
    def __init__(self, tracks: list[Track]):
        self.X = []
        self.y = []

        genres = sorted(set(t.genre for t in tracks))
        genre_to_idx = {g: i for i, g in enumerate(genres)}

        for t in tracks:
            self.X.append(extract_feature_vector(t.features))
            self.y.append(genre_to_idx[t.genre])

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32) # float64 to match extract_feature_vector 
        self.y = torch.tensor(np.array(self.y), dtype=torch.long)

    def __len__(self):
        assert len(self.x) == len(self.y)
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TrackToGenreMLP(nn.Module):
    """
    PyTorch `nn.Module` to learn to classify tracks by genre.
    
    Referenced https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html#define-the-class
    for building the mlp
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)