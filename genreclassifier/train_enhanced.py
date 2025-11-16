"""
Enhanced training module with feature normalization, improved architecture,
and comprehensive training utilities.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Tuple, Dict, Optional
from collections import Counter
from dataclasses import replace

from .features import Track, NORMALIZE_MASK


def combine_rare_genres(tracks: list[Track], min_samples: int = 20) -> list[Track]:
    """
    Combine genres with fewer than min_samples into an 'Other' category.
    This helps with class imbalance.
    """
    genre_counts = Counter(t.genre for t in tracks)
    rare_genres = {g for g, count in genre_counts.items() if count < min_samples}
    
    if not rare_genres:
        return tracks
    
    print(f"Combining {len(rare_genres)} rare genres into 'Other': {sorted(rare_genres)}")
    
    combined_tracks = []
    for track in tracks:
        if track.genre in rare_genres:
            # Create new track with 'Other' genre
            combined_tracks.append(replace(track, genre='Other'))
        else:
            combined_tracks.append(track)
    
    return combined_tracks

def make_oversampling_sampler(dataset):
    """
    Create a WeightedRandomSampler that oversamples minority classes.
    Weighted by inverse class frequency.
    """
    labels = dataset.y.numpy()
    class_sample_count = np.array([np.sum(labels == c) for c in np.unique(labels)])
    
    # inverse frequency
    class_weights = 1. / class_sample_count
    
    # assign weight to each sample
    sample_weights = np.array([class_weights[label] for label in labels])
    sample_weights = torch.from_numpy(sample_weights).double()
    
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


class NormalizedTrackDataset(Dataset):
    """
    Enhanced PyTorch dataset with feature normalization and genre mapping.
    """
    def __init__(
        self, 
        tracks: list[Track], 
        scaler: Optional[StandardScaler] = None, 
        genre_to_idx: Optional[Dict[str, int]] = None
    ):
        self.X = []
        self.y = []

        # Create or use genre mapping
        if genre_to_idx is None:
            genres = sorted(set(t.genre for t in tracks))
            self.genre_to_idx = {g: i for i, g in enumerate(genres)}
            self.idx_to_genre = {i: g for g, i in self.genre_to_idx.items()}
        else:
            self.genre_to_idx = genre_to_idx
            self.idx_to_genre = {i: g for g, i in genre_to_idx.items()}

        # Extract feature vectors
        for t in tracks:
            self.X.append(t.features.extract_feature_vector())
            self.y.append(self.genre_to_idx[t.genre])

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.int64)

        self.normalize_mask = NORMALIZE_MASK

        # Fit scaler only on masked continuous features
        if scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(self.X[:, self.normalize_mask])
        else:
            self.scaler = scaler

        # Apply normalization
        X_normalized = self.X.copy()
        X_normalized[:, self.normalize_mask] = self.scaler.transform(
            self.X[:, self.normalize_mask]
        )

        # Convert to tensors
        self.X = torch.tensor(X_normalized, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

        self.num_classes = len(self.genre_to_idx)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalanced data using inverse frequency."""
        class_counts = Counter(self.y.numpy())
        total = len(self.y)
        freq = np.array([class_counts[i] for i in range(self.num_classes)])
        weights = 1 / np.log(1.2 + freq)
        weights = torch.tensor(weights / weights.mean(), dtype=torch.float32)
        return weights



class EnhancedTrackToGenreMLP(nn.Module):
    """
    Enhanced MLP with dropout, batch normalization, and better architecture.
    """
    def __init__(
        self, 
        input_dim: int, 
        num_classes: int, 
        hidden_dims: list = [512, 256, 128], #[256, 128, 64], 
        dropout: float = 0.15,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            linear = nn.Linear(prev_dim, hidden_dim)
            # Better weight initialization
            nn.init.kaiming_normal_(linear.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(linear.bias, 0)
            layers.append(linear)
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer with better initialization
        output_layer = nn.Linear(prev_dim, num_classes)
        nn.init.xavier_normal_(output_layer.weight)
        nn.init.constant_(output_layer.bias, 0)
        layers.append(output_layer)
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: torch.device, 
                max_grad_norm: float = 1.0) -> Tuple[float, float]:
    """Train for one epoch with gradient clipping."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
             device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds) * 100  # Convert to percentage
    
    return avg_loss, accuracy, np.array(all_labels), np.array(all_preds)


def train_model(tracks: list[Track], 
                train_split: float = 0.7,
                val_split: float = 0.15,
                test_split: float = 0.15,
                batch_size: int = 32,
                num_epochs: int = 200,
                learning_rate: float = 5e-4,
                hidden_dims: list = [384, 192, 96],
                dropout: float = 0.35,
                use_class_weights: bool = True,
                early_stopping_patience: int = 20,
                combine_rare: bool = True,
                min_samples_per_genre: int = 15,
                device: Optional[torch.device] = None) -> Dict:
    """
    Complete training pipeline with train/val/test split, early stopping, and evaluation.
    
    Args:
        combine_rare: If True, combine genres with < min_samples_per_genre into 'Other'
        min_samples_per_genre: Minimum samples required to keep a genre separate
    
    Returns:
        Dictionary containing:
        - model: trained model
        - train_dataset: training dataset (with scaler and genre mappings)
        - val_dataset: validation dataset
        - test_dataset: test dataset
        - history: training history (losses and accuracies)
        - test_metrics: final test set metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Combine rare genres to reduce class imbalance
    if combine_rare:
        tracks = combine_rare_genres(tracks, min_samples=min_samples_per_genre)
    

    full_dataset = NormalizedTrackDataset(tracks)
    input_dim = full_dataset.X.shape[1]
    num_classes = full_dataset.num_classes


    print(f"Total samples: {len(full_dataset)}")
    print(f"Number of classes: {full_dataset.num_classes}")
    print(f"Input dimension: {input_dim}")
    print(f"Classes: {list(full_dataset.genre_to_idx.keys())}")

    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    train_subset, val_subset, test_subset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Making sure to use the correct indices for the train, val, and test datasets
    train_indices = train_subset.indices
    val_indices = val_subset.indices
    test_indices = test_subset.indices

    train_dataset = NormalizedTrackDataset(
        [tracks[i] for i in train_indices],
        scaler=full_dataset.scaler,
        genre_to_idx=full_dataset.genre_to_idx
    )

    val_dataset = NormalizedTrackDataset(
        [tracks[i] for i in val_indices],
        scaler=full_dataset.scaler,
        genre_to_idx=full_dataset.genre_to_idx
    )

    test_dataset = NormalizedTrackDataset(
        [tracks[i] for i in test_indices],
        scaler=full_dataset.scaler,
        genre_to_idx=full_dataset.genre_to_idx
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    sampler = make_oversampling_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = EnhancedTrackToGenreMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout
    ).to(device)
    
    # Loss function with class weights
    if use_class_weights:
        class_weights = train_dataset.get_class_weights().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using class weights for imbalanced data")
        print(f"Class weights: {dict(zip(train_dataset.idx_to_genre.values(), class_weights.cpu().numpy()))}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # OneCycle learning rate schedule (stepped per batch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate * 10,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader)
    )

    # Tracking
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_score = -1
    patience_counter = 0
    best_model_state = None

    print("\nStarting training...")
    for epoch in range(num_epochs):

        # Training
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * X_batch.size(0)
            _, predicted = preds.max(1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # Validation
        val_loss, val_acc, val_labels, val_preds = evaluate(model, val_loader, criterion, device)
        val_macro_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)

        # Combined validation score
        combined_score = 0.5 * val_macro_f1 + 0.5 * (val_acc / 100.0)

        # Early stopping check
        improved = combined_score > best_score
        if improved:
            best_score = combined_score
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Logging
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]['lr']
            flag = "✓" if improved else " "
            print(
                f"Epoch {epoch+1}/{num_epochs} (LR: {lr_now:.6f}) {flag}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                f"MacroF1: {val_macro_f1:.4f}, Combined: {combined_score:.4f}"
            )


    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_labels, test_preds = evaluate(model, test_loader, criterion, device)
    
    # Calculate detailed metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average='weighted', zero_division=0
    )
    cm = confusion_matrix(test_labels, test_preds)
    
    test_metrics = {
        'loss': test_loss,
        'accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.2f}%")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    return {
        'model': model,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'history': history,
        'test_metrics': test_metrics,
        'genre_to_idx': train_dataset.genre_to_idx,
        'idx_to_genre': train_dataset.idx_to_genre
    }

