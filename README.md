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
- umap-learn

### Installation

Conda environment:
```bash
conda create --name music python=3.11
conda activate music
pip install -r requirements.txt
```

### Data Setup

**NOTE: FOR EASE OF USAGE, you do NOT need to set the datasets up manually to try out the model. We have saved pkl files so that you can run training without setting up the dataset on your computer.**

In the folder containing this repo (parent directory), download the following files:
- [CD1 Genre Ground Truth](https://www.tagtraum.com/msd_genre_datasets.html) - save as `msd_tagtraum_cd1.cls`
- [Lakh Midi Kaggle Dataset](https://www.kaggle.com/datasets/nddimension/lmd-matched?resource=download) - extract to `lmd_matched/`
- [Match Scores](https://colinraffel.com/projects/lmd/) - save as `match_scores.json`
  - A JSON file which lists the match confidence score for every match in LMD-matched and LMD-aligned.

### Expected file structure:

```
parent-folder/
├── MidiGenreClassifier/  (this repo)
├── MidiFiles/            (MIDI files)
├── match_scores.json     (matching file)
└── msd_tagtraum_cd1.cls  (genre labels)
```

## Usage

### GUI Interface tool

```bash
# Activate conda environment
conda activate music
# Run GUI
python main.py
```

### Training the Model

```bash
# Activate conda environment
conda activate music
# Train model
python run_training.py
```

### Features

For more information on how features were extracted from MIDI files, view `genreclassifier/features.py`.


## Acknowledgements

This work used the following datasets:
- **Lakh MIDI dataset**
  - A description can be found [here](https://colinraffel.com/projects/lmd/).
  - We utilized a subset of the Lakh MIDI dataset from Kaggle, aligned to entries in the [Million Song Dataset](http://millionsongdataset.com/)
- **Tagtraum genre dataset**
  - A description and download link can be found [here](https://www.tagtraum.com/msd_genre_datasets.html)

