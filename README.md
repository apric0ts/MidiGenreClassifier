# MidiGenreClassifier
Fall 2025 CS 4100 - Artificial Intelligence Final Project


## Setup

### Dependencies

See `requirements.txt` for full package brevity

Packages:
- numpy
- pandas
- matplotlib
- pretty_midi
- pytorch



Create a conda env or python env:
```bash
conda create --name music
```

(untested)
```bash
python3 -m venv music
source music/bin/activate
```

Navigate to this repo's directory, and:
```
pip install -r requirements.txt
```

In the folder containing this repo, download the following files:
- [CD1 Genre Ground Truth](https://www.tagtraum.com/msd_genre_datasets.html)
- [Lakh Midi Kaggle Dataset](https://www.kaggle.com/datasets/imsparsh/lakh-midi-clean?resource=download)
- [Match Scores](https://colinraffel.com/projects/lmd/)
  - A json file which lists the match confidence score for every match in LMD-matched and LMD-aligned.

## Acknowledgements

This work used the following datasets:
- Lakh MIDI dataset
  - A description can be found [here](https://colinraffel.com/projects/lmd/).
  - We utilized a subset of the Lakh MIDI dataset from Kaggle, aligned to entries in the [Million Song Dataset ](http://millionsongdataset.com/)
https://www.tagtraum.com/msd_genre_datasets.html
- tagtraum genre dataset
  - A description and download link can be found [here](https://www.tagtraum.com/msd_genre_datasets.html)


