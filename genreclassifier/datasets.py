import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from pathlib import Path
from typing import Generator

from tqdm import tqdm
import pretty_midi
from .features import *
from . import utils 


def get_all_track_information(
    midi_files_path: Path | str,
    match_scores_path: Path | str,
    genres_path: Path | str,
    cache_path: Path | str | None = None,
    files_walked_count: int | None = None,
    chunk_size: int = 1000,
) -> list[Track]:

    # If cache exists → load it normally
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

    midi_features = dict(_extract_midi_files(
        midi_files_path=midi_files_path,
        match_scores_path=match_scores_path,
        limit=files_walked_count,
        valid_track_ids=valid_track_ids
    ))

    # ---- NEW: streaming chunk writer ----
    buffer = []
    chunk_index = 0
    base_path = str(cache_path).replace(".pkl", "") if cache_path else None

    def flush_buffer():
        nonlocal chunk_index
        if not buffer or not base_path:
            return
        out_path = f"{base_path}_{chunk_index}.pkl"
        utils._cache_pickle(buffer, out_path)
        print(f"Saved chunk {chunk_index} with {len(buffer)} tracks")
        buffer.clear()
        chunk_index += 1

    # ---- iterate and stream-save ----
    for track_id, genres in _extract_genres(genres_path):
        if track_id in midi_features:
            for genre in genres:
                buffer.append(
                    Track(
                        track_id=track_id,
                        genre=genre,
                        features=midi_features[track_id],
                    )
                )

                # flush memory every chunk_size
                if len(buffer) >= chunk_size:
                    flush_buffer()

    # flush remainder
    flush_buffer()

    print("Finished streaming and saving track chunks.")

    # Return nothing (or paths), because loading later is cheaper
    return []

###
# Helpers
###

def extract_data_from_midi(midi_file_path: Path | str):
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)

    # Extract list of instruments from the `pretty_midi.Instrument`
    instruments: list[int] = [data.program for data in midi_data.instruments]
    pretty_midi.Instrument

    # Extract solely the key signatures from the `pretty_midi.KeySignature` 
    key_signatures: list[int] = [data.key_number for data in midi_data.key_signature_changes]

    # Extract solely the time signature from the `pretty_midi.TimeSignature`
    time_signatures: list[tuple[int, int]] = [(data.numerator, data.denominator) for data in midi_data.time_signature_changes]

    lyrics: bool = len(midi_data.lyrics) > 0

    length = midi_data.get_end_time()

    rhythm = compute_rhythm_features(midi_data, length)
    harmony = compute_harmony_features(midi_data)
    melody = compute_melody_features(midi_data, length)
    instr_strength = compute_instrument_strength(midi_data)
    structure = compute_structure_features(midi_data)

    features = MidiFeatures(
        tempo = midi_data.estimate_tempo(),
        instruments = instruments,
        key_signatures = key_signatures,
        time_signatures = time_signatures,
        lyrics = lyrics,
        length = length,
        rhythm = rhythm,
        harmony = harmony,
        melody = melody,
        instr_strength = instr_strength,
        structure = structure,
    )
    return features



def _midi_worker(job):
    """Top-level function so it can be pickled for multiprocessing."""
    track_id, full_path = job
    try:
        features = extract_data_from_midi(full_path)
        return (track_id, features)
    except Exception:
        return None


def _extract_midi_files(
    midi_files_path: Path | str,
    match_scores_path: Path | str,
    valid_track_ids: list,
    limit: int | None = None,
):
    """
    Extracts features from MIDI files located in the LMD-matched directory structure.
    """

    print("Analyzing LMD-matched MIDI files...")
    assert valid_track_ids is not None
    valid_track_ids = set(valid_track_ids)

    jobs = []
    files_walked = 0

    # Collect jobs
    for root, dirs, files in os.walk(midi_files_path):
        for file in files:

            if not (file.endswith(".mid") or file.endswith(".midi")):
                continue

            if limit is not None and files_walked >= limit:
                break

            full_path = os.path.join(root, file)
            track_id = os.path.basename(os.path.dirname(full_path))

            if not track_id.startswith("TR"):
                continue

            if track_id not in valid_track_ids:
                continue

            jobs.append((track_id, full_path))
            files_walked += 1

    print(f"Queued {len(jobs)} MIDI files for multiprocessing...")


    # Execute all the jobs, multiprocess
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_midi_worker, job) for job in jobs]

        for f in tqdm(as_completed(futures), total=len(futures)):
            result = f.result()
            if result is not None:
                yield result

    print(f"Walked {len(jobs)} MIDI files.")



def _extract_genres(genres_path: Path | str):
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