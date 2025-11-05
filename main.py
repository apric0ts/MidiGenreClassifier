
import os
import genreclassifier as gc

import numpy as np

from collections import Counter

if __name__ == "__main__":

    """
    Example project structure:

    outer-folder
    - MidiGenreClassifier (our repo)
    - MidiFiles (self explanatory)
    - matches_scores.json (matching midi files to tracks in MSD)
    - msd_tagtraum_cd1.cls (genres)
    """
    

    # if we execute this from the project root:

    # Project root directory
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    midi_files_path = os.path.join(parent_dir, "MidiFiles")
    match_scores_path = os.path.join(parent_dir, "match_scores.json")
    genres_path = os.path.join(parent_dir, "msd_tagtraum_cd1.cls")


    all_track_info: list[gc.Track] = gc.get_all_track_information(midi_files_path, match_scores_path, genres_path)
    

    # Exmaples of how to use:
    tempos = np.array([track.features.tempo for track in all_track_info])

    print(tempos.min(), tempos.max(), tempos.mean())

    genres = Counter([track.genre for track in all_track_info])
    print(genres)

    
        