"""
Embedding Visualization Script
Generates t-SNE and UMAP plots for:
1. Raw MIDI features
2. Model-learned embeddings (final hidden layer)
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from torch.utils.data import DataLoader

def extract_embeddings(model, dataloader, device):
    """
    Pass data through model and extract final hidden-layer embeddings.
    """
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)

            # Forward pass until last layer
            x = X_batch
            for layer in list(model.children())[:-1]:  # skip output layer
                x = layer(x)

            embeddings.append(x.cpu().numpy())
            labels.append(y_batch.numpy())

    return np.vstack(embeddings), np.hstack(labels)


def run_embedding_visualizations(model, dataset, genre_names, device):
    """
    Run t-SNE + UMAP on raw features and learned embeddings.
    """
    print("\n" + "="*60)
    print("RUNNING t-SNE & UMAP VISUALIZATION")
    print("="*60)

    os.makedirs("visualizations/embeddings", exist_ok=True)

    # Load data
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    raw_features = dataset.X  # numpy array (already scaled)
    raw_labels = dataset.y

    print(f"Raw Feature Shape: {raw_features.shape}")

    # t-SNE on RAW FEATURES
    print("Running t-SNE on raw features...")
    tsne_raw = TSNE(n_components=2, perplexity=35, learning_rate=200).fit_transform(raw_features)

    plt.figure(figsize=(10, 8))
    for idx, g in enumerate(genre_names):
        mask = raw_labels == idx
        plt.scatter(tsne_raw[mask, 0], tsne_raw[mask, 1], s=12, alpha=0.6, label=g)

    plt.legend()
    plt.title("t-SNE Projection of RAW MIDI Features")
    plt.savefig("visualizations/embeddings/tsne_raw.png", dpi=300)
    plt.close()
    print("Saved t-SNE raw → visualizations/embeddings/tsne_raw.png")

    # UMAP on RAW FEATURES
    print("Running UMAP on raw features...")
    umap_raw = umap.UMAP(n_neighbors=30, min_dist=0.1).fit_transform(raw_features)

    plt.figure(figsize=(10, 8))
    for idx, g in enumerate(genre_names):
        mask = raw_labels == idx
        plt.scatter(umap_raw[mask, 0], umap_raw[mask, 1], s=12, alpha=0.6, label=g)

    plt.legend()
    plt.title("UMAP Projection of RAW MIDI Features")
    plt.savefig("visualizations/embeddings/umap_raw.png", dpi=300)
    plt.close()
    print("Saved UMAP raw → visualizations/embeddings/umap_raw.png")

    # Get Model Embeddings
    print("Extracting model-learned embeddings...")
    learned_embeddings, embed_labels = extract_embeddings(model, loader, device)
    print(f"Learned Embeddings Shape: {learned_embeddings.shape}")

    # t-SNE on LEARNED EMBEDDINGS
    print("Running t-SNE on learned embeddings...")
    tsne_embed = TSNE(n_components=2, perplexity=35, learning_rate=200).fit_transform(learned_embeddings)

    plt.figure(figsize=(10, 8))
    for idx, g in enumerate(genre_names):
        mask = embed_labels == idx
        plt.scatter(tsne_embed[mask, 0], tsne_embed[mask, 1], s=12, alpha=0.6, label=g)

    plt.legend()
    plt.title("t-SNE Projection of Model Embeddings")
    plt.savefig("visualizations/embeddings/tsne_embeddings.png", dpi=300)
    plt.close()
    print("Saved t-SNE embeddings → visualizations/embeddings/tsne_embeddings.png")

    # UMAP on LEARNED EMBEDDINGS
    print("Running UMAP on learned embeddings...")
    umap_embed = umap.UMAP(n_neighbors=30, min_dist=0.1).fit_transform(learned_embeddings)

    plt.figure(figsize=(10, 8))
    for idx, g in enumerate(genre_names):
        mask = embed_labels == idx
        plt.scatter(umap_embed[mask, 0], umap_embed[mask, 1], s=12, alpha=0.6, label=g)

    plt.legend()
    plt.title("UMAP Projection of Model Embeddings")
    plt.savefig("visualizations/embeddings/umap_embeddings.png", dpi=300)
    plt.close()

    print("Saved UMAP embeddings → visualizations/embeddings/umap_embeddings.png")

    print("\nAll embedding visualizations completed!")
