from pathlib import Path
import os
import pickle
import json
import hashlib

def _cache_pickle(obj, path: Path | str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def _load_pickle(path: Path | str):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

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