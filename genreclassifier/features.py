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