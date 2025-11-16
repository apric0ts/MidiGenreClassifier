# MidiGenreClassifier
Fall 2025 CS 4100 - Artificial Intelligence Final Project

A neural network-based genre classification system for MIDI files using PyTorch.

## Setup

### Dependencies

See `requirements.txt` for the full package list.

Key packages:
- numpy
- pandas
- matplotlib
- pretty_midi
- pytorch
- scikit-learn

### Installation

**Option 1: Python venv**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Option 2: Conda venv**
```bash
conda create --name music python=3.11
conda activate music
pip install -r requirements.txt
```

### Data Setup

In the folder containing this repo (parent directory), download the following files:
- [CD1 Genre Ground Truth](https://www.tagtraum.com/msd_genre_datasets.html) → save as `msd_tagtraum_cd1.cls`
- [Lakh Midi Kaggle Dataset](https://www.kaggle.com/datasets/nddimension/lmd-matched?resource=download) → extract to `lmd_matched/`
- [Match Scores](https://colinraffel.com/projects/lmd/) → save as `match_scores.json`
  - A JSON file which lists the match confidence score for every match in LMD-matched and LMD-aligned.

**Expected directory structure:**
```
parent-folder/
├── MidiGenreClassifier/  (this repo)
├── MidiFiles/            (MIDI files)
├── match_scores.json     (matching file)
└── msd_tagtraum_cd1.cls  (genre labels)
```

## Usage

### Training the Model

Run the enhanced training pipeline:

```bash
# Activate virtual environment
source venv/bin/activate

# Run training
python run_training.py
```

The training script will:
- Load MIDI tracks (uses cache if available: `midi_features_cache_large.pkl`)
- Display genre distribution
- Train the model with:
  - Feature normalization
  - Enhanced MLP architecture (384→192→96)
  - Dropout and batch normalization
  - Class weighting for imbalanced data
  - Early stopping based on validation accuracy
- Display final test set metrics
- Optionally save the trained model

### Features

The model extracts 157-dimensional feature vectors from MIDI files:
- **5 continuous features**: tempo, length, lyrics (binary), average time signature
- **128 one-hot instrument features**: MIDI program IDs (0-127)
- **24 one-hot key signature features**: Major/minor keys (0-23)

### Model Architecture

- **Enhanced MLP**: 3 hidden layers (384→192→96) with ReLU activations
- **Regularization**: Dropout (0.35), batch normalization, L2 weight decay
- **Training**: Adam optimizer with learning rate scheduling, gradient clipping
- **Class imbalance handling**: Inverse frequency weighting, rare genre combination

## Project Structure

```
MidiGenreClassifier/
├── genreclassifier/          # Main package
│   ├── __init__.py
│   ├── datasets.py          # Data loading and caching
│   ├── features.py          # Feature extraction from MIDI
│   ├── train_enhanced.py    # Enhanced training pipeline
│   └── utils.py             # Utility functions
├── run_training.py           # Main training script
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Package configuration
├── README.md                 # This file
└── midi_features_cache_large.pkl  # Cached features (auto-generated)
```

## Acknowledgements

This work used the following datasets:
- **Lakh MIDI dataset**
  - A description can be found [here](https://colinraffel.com/projects/lmd/).
  - We utilized a subset of the Lakh MIDI dataset from Kaggle, aligned to entries in the [Million Song Dataset](http://millionsongdataset.com/)
- **Tagtraum genre dataset**
  - A description and download link can be found [here](https://www.tagtraum.com/msd_genre_datasets.html)

