from dataclasses import dataclass

import numpy as np
import pretty_midi

PROGRAM_ID_TO_INSTRUMENT_NAME = {
    i: pretty_midi.program_to_instrument_name(i)
    for i in range(128)
}

@dataclass
class MidiFeatures:
    """
    Features of a MIDI File. Easily extendible to include other types of features.

    tempo: estimated tempo of the song
    instruments: list of instruments (represented as a list of ints)
    key_signatures: list of key signatures throughout the song (represented as a list of ints)
    time_signatures: list of time signatures throughout the song
    lyrics: whether the song has lyrics or not
    length: (rounded) length of the song in seconds
    """
    tempo: np.float64
    instruments: list[np.int64]
    key_signatures: list[int]
    time_signatures: list[tuple[int, int]]
    lyrics: bool
    length: np.float64



@dataclass
class Track:
    """
    Contains information about a specific track
    """
    track_id: str
    genre: str
    features: MidiFeatures


def extract_feature_vector(features: MidiFeatures) -> np.ndarray:
    """
    Turns `MidiFeatures` into an `np.ndarray` that can be fed into a NN for training.

    Returns a vector of length 5 + 128 + 24 = 157

    A design choice here is to use separate values for major and minor of the same key, i.e.
    C major and C minor are represented differently as opposed to the same key. If our sample size is too small,
    (perhaps there are very few songs that have C minor), then we can configure this encoding such that 
    C major and C minor map to the same feature (so `key_vec = np.zeros(12), key_vec[k % 12] = 1.0`)

    The final vector is of form: [tempo, length, lyrics, avg_num, avg_den, *instrument_vec, *key_vec],
    where each element in the vector is of type `np.float64`
    """
    tempo = features.tempo
    length = features.length
    lyrics = 1.0 if features.lyrics else 0.0

    # Instruments
    # Convert list of ints representing instruments into a (one-hot) vector of length 128
    instrument_vec = np.zeros(128)
    for i in features.instruments:
        instrument_vec[i] = 1.0

    # Key signatures
    # Convert list of ints representing key signatures into all possible keys

    key_vec = np.zeros(24)
    for k in features.key_signatures:
        key_vec[k] = 1.0

    # Time signatures
    # encode as average numerator/denominator over the whole piece (don't know if this is the best thing to do
    # but I'd assume the time signatures don't change much over the whole piece?)
    if features.time_signatures:
        numerators, denominators = zip(*features.time_signatures)
        avg_num = np.mean(numerators)
        avg_den = np.mean(denominators)
    else: # if there's no valid key signatures found from the midi file
        avg_num, avg_den = 4, 4  # default common time

    # combine vectors, return
    return np.concatenate([
        [tempo, length, lyrics, avg_num, avg_den],
        instrument_vec,
        key_vec,
    ])
