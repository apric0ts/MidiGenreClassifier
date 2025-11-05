import os
import json
import hashlib
from pathlib import Path
from typing import Generator

from tqdm import tqdm
import pretty_midi

from .features import Track, MidiFeatures
from .utils import _load_pickle, _cache_pickle


def get_all_track_information(
    midi_files_path: Path | str,
    match_scores_path: Path | str,
    genres_path: Path | str,
    cache_path: Path | str = "midi_features_cache.pkl",
) -> list[Track]:

    cached = _load_pickle(cache_path)
    if cached is not None:
        print(f"Loaded {len(cached)} tracks from cache")
        return cached

    # Features dict
    midi_features = dict(_extract_midi_files(midi_files_path, match_scores_path))

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

    _cache_pickle(tracks, cache_path)
    print(f"Saved {len(tracks)} tracks to cache: {cache_path}")


    return tracks

###
# Helpers
###

def _md5(path: Path | str) -> str:
    # use chunked reading to reduce memory use
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):  # 1 MB chunks
            h.update(chunk)
    return h.hexdigest()


def _md5_to_track_id(path: Path | str) -> dict[str, str]:

    with open(path) as f:
        match_data = json.load(f)

    md5_to_msd = {}
    for msd_id, midis in match_data.items():
        for midi_md5 in midis:
            md5_to_msd[midi_md5] = msd_id

    return md5_to_msd



def _extract_midi_files(
    midi_files_path: Path | str,
    match_scores_path: Path | str,
    limit: int = 500,
) -> Generator[str, MidiFeatures]:
    """
    Extracts features from MIDI file in the given directory
    https://www.kaggle.com/datasets/imsparsh/lakh-midi-clean?resource=download
    """

    md5_to_track_id: dict[str, str] = _md5_to_track_id(match_scores_path)
    known_md5s = set(md5_to_track_id.keys())  # for O(1) lookups

    files_walked: int = 0

    for root, dirs, files in tqdm(os.walk(midi_files_path)):
        for file in files:
            if files_walked >= limit:
                return
            if file.endswith("midi") or file.endswith("mid"):
                full_path = os.path.join(root, file)
                try:
                    md5 = _md5(full_path)
                    if md5 not in known_md5s:
                        continue

                    midi_data = pretty_midi.PrettyMIDI(full_path)

                    # Extract solely the key signatures from the `pretty_midi.KeySignature`
                    key_signatures: list[int] = [data.key_number for data in midi_data.key_signature_changes]

                    # Extract solely the time signature from the `pretty_midi.TimeSignature`
                    time_signatures: list[tuple[int, int]] = [(data.numerator, data.denominator) for data in midi_data.time_signature_changes]

                    lyrics: bool = len(midi_data.lyrics) > 0

                    features = MidiFeatures(
                        tempo = midi_data.estimate_tempo(),
                        key_signatures = key_signatures,
                        time_signatures = time_signatures,
                        lyrics = lyrics,
                        length = midi_data.get_end_time()
                    )

                    track_id = md5_to_track_id[md5]

                    files_walked += 1
                    yield track_id, features

                except Exception as e:
                    pass
                


def _extract_genres(genres_path: Path | str) -> Generator[str, list[str]]:
    """
    Extracts genres from the CD1 Genre Ground Truth dataset from https://www.tagtraum.com/msd_genre_datasets.html
    at the given path on local machine
    """

    track_id_to_genres: dict[str, list[str]] = {}
    malformed_dataset_lines: list[str] = []

    # Open cd1 dataset
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
