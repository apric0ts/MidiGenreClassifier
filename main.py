
import os
from collections import Counter

import numpy as np

import genreclassifier as gc


if __name__ == "__main__":

    """
    Example project structure:

    outer-folder
    - MidiGenreClassifier (our repo)
    - MidiFiles (self explanatory)
    - matches_scores.json (matching midi files to tracks in MSD)
    - msd_tagtraum_cd1.cls (genres)
    """
    # if we execute this from the repo root:

    # Project root directory
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    midi_files_path = os.path.join(parent_dir, "lmd_matched")
    match_scores_path = os.path.join(parent_dir, "match_scores.json")
    genres_path = os.path.join(parent_dir, "msd_tagtraum_cd1.cls")


    tracks: list[gc.Track] = gc.get_all_track_information(
        midi_files_path, 
        match_scores_path, 
        genres_path, 
        cache_path = "midi_features_cache.pkl",
        files_walked_count=None # change this number to adjust the breadth of midi files we're analyzing. None means get all info
    )
    

    # Examples of how to use:

    # Get min, max, mean of all track tempos
    tempos = np.array([track.features.tempo for track in tracks])
    print(tempos.min(), tempos.max(), tempos.mean())

    # Get number of instances of each genre across all tracks
    genres = Counter([track.genre for track in tracks])
    print(genres)
    print(f"There are {len(genres)} unique genres across these tracks")


    # Get number of instances of each instrument across all tracks
    instrument_ids = []
    for track in tracks:
        instrument_ids.extend([int(x) for x in track.features.instruments]) # instruments represented as np.int64

    instruments = Counter([gc.PROGRAM_ID_TO_INSTRUMENT_NAME[id] for id in instrument_ids])
    print(instruments)
    print(f"There are {len(instruments)} unique instruments across these tracks")
    
    
    # Try out the nn stuff:
    dataset = gc.TrackDataset(tracks)

    input_dim = dataset.X.shape[1] # dataset.X is [num_tracks, num_features]
    model = gc.TrackToGenreMLP(input_dim=input_dim, num_classes=len(genres))