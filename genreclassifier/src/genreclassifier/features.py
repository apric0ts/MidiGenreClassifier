from dataclasses import dataclass


@dataclass
class MidiFeatures:
    """
    From a `PrettyMIDI` object, we can extract the following,
    as well as some other information:

    tempo: estimated tempo of the song
    key_signature_changes: list of key signatures throughout the song
    time_signature_changes: list of time signatures throughout the song
    lyrics: whether the song has lyrics or not
    length: (rounded) length of the song in seconds
    """
    tempo: int
    key_signatures: list[str]
    time_signatures: list[str]
    lyrics: bool
    length: int



@dataclass
class Track:
    """
    Contains information about a specific track
    """
    track_id: str
    genre: str
    features: str


def _hash_instruments():
    """
    Give each instrument name a unique integer for model training
    """
    