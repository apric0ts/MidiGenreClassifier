from pathlib import Path
import os
import pickle
import json
import hashlib

def _cache_pickle(obj, path: Path | str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def _load_pickle(base_path):
    """
    Load chunked pickle files: base_path_0.pkl, base_path_1.pkl, ...
    Returns combined list.
    """
    import pickle
    from glob import glob

    base = str(base_path).replace(".pkl", "")
    files = sorted(glob(base + "_*.pkl"))

    if not files:
        return None

    all_data = []
    for fp in files:
        with open(fp, "rb") as f:
            all_data.extend(pickle.load(f))

    print(f"Loaded {len(all_data)} items from {len(files)} chunked pkls")
    return all_data
