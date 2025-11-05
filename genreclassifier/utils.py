from pathlib import Path
import os
import pickle

def _cache_pickle(obj, path: Path | str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def _load_pickle(path: Path | str):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None
