import os

from pathlib import Path
from typing import Generator

from tqdm import tqdm
import pretty_midi
from .features import Track, MidiFeatures
from . import utils 


def get_all_track_information(
    midi_files_path: Path | str,
    match_scores_path: Path | str,
    genres_path: Path | str,
    cache_path: Path | str | None = None,
    files_walked_count: int | None = None
) -> list[Track]:
    """
    Get a list of tracks, with associated genres, from the given local paths.

    Options available to:
    - cache the saved tracks from previous runs, if the local files have already
    been analyzed once
    - limit the number of files counted
    """

    if cache_path:
        cached = utils._load_pickle(cache_path)
        if cached is not None:
            print(f"Loaded {len(cached)} tracks from cache")
            return cached

    valid_track_ids = set()
    with open(genres_path) as f:
        for line in f:
            if not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 2:
                    valid_track_ids.add(parts[0])

    # Features dict
    midi_features = dict(_extract_midi_files(
        midi_files_path=midi_files_path,
        match_scores_path=match_scores_path,
        limit=files_walked_count,
        valid_track_ids=valid_track_ids
    ))

    # Iterate over features dict, creating tracks
    tracks: list[Track] = []
    for track_id, genres in _extract_genres(genres_path):
        if track_id in midi_features:
            for genre in genres:  # multiple possible genres
                tracks.append(
                    Track(
                        track_id=track_id,
                        genre=genre,
                        features=midi_features[track_id],
                    )
                )

    if cache_path:
        utils._cache_pickle(tracks, cache_path)
        print(f"Saved {len(tracks)} tracks to cache: {cache_path}")


    return tracks

###
# Helpers
###

def _extract_data_from_midi(midi_file_path: Path | str):
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)

    # Extract list of instruments from the `pretty_midi.Instrument`
    instruments: list[int] = [data.program for data in midi_data.instruments]
    pretty_midi.Instrument

    # Extract solely the key signatures from the `pretty_midi.KeySignature` 
    key_signatures: list[int] = [data.key_number for data in midi_data.key_signature_changes]

    # Extract solely the time signature from the `pretty_midi.TimeSignature`
    time_signatures: list[tuple[int, int]] = [(data.numerator, data.denominator) for data in midi_data.time_signature_changes]

    lyrics: bool = len(midi_data.lyrics) > 0

    features = MidiFeatures(
        tempo = midi_data.estimate_tempo(),
        instruments = instruments,
        key_signatures = key_signatures,
        time_signatures = time_signatures,
        lyrics = lyrics,
        length = midi_data.get_end_time()
    )
    return features


def _extract_midi_files(
    midi_files_path: Path | str,
    match_scores_path: Path | str,
    valid_track_ids: list,
    limit: int | None = None,
) -> Generator[str, MidiFeatures]:
    """
    Extracts features from MIDI files located in the LMD-matched directory structure.

    Expected directory pattern:
    lmd_matched/X/Y/Z/TRXXXX.../
        ├── file1.mid
        ├── file2.mid
        └── ...

    Where the folder name 'TRXXXX...' is the MSD track_id.
    """

    files_walked: int = 0

    print("Analyzing LMD-matched MIDI files...")
    assert valid_track_ids is not None

    for root, dirs, files in tqdm(os.walk(midi_files_path)):
        for file in files:
            if not (file.endswith(".mid") or file.endswith(".midi")):
                continue

            if limit is not None and files_walked >= limit:
                print(f"Reached file limit: {limit}")
                return

            full_path = os.path.join(root, file)

            try:
                track_id = os.path.basename(os.path.dirname(full_path))

                if not track_id.startswith("TR"):
                    continue

                if track_id not in valid_track_ids:
                    continue

                features = _extract_data_from_midi(full_path)

                files_walked += 1
                yield track_id, features

            except Exception:
                continue

    print(f"Walked {files_walked} MIDI files.")



def _extract_genres(genres_path: Path | str) -> Generator[str, list[str]]:
    """
    Extracts genres from the CD1 Genre Ground Truth dataset from https://www.tagtraum.com/msd_genre_datasets.html
    at the given path on local machine
    """

    malformed_dataset_lines: list[str] = []

    # Open cd1 dataset
    print("Extracting genres...")
    with open(genres_path) as f:
        for line in tqdm(f.readlines()):
            # Only parse lines that are not commented out
            if line[0] == "#":
                continue

            # Lines are formatted like so:
            # TRACK_ID GENRE1
            # TRACK_ID GENRE1 GENRE2
            split_lines = line.split()
            track = split_lines[0]
            genres: list[str] = split_lines[1:] # a list of 

            assert len(split_lines) >= 1

            try:
                assert len(genres) >= 1, "This track has no genres"
            except AssertionError:
                malformed_dataset_lines.append(track)
                continue

            yield track, genres

    print(f"Malformed labels: {malformed_dataset_lines}")
