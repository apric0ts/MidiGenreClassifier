from dataclasses import dataclass

import numpy as np
import pretty_midi

PROGRAM_ID_TO_INSTRUMENT_NAME = {
    i: pretty_midi.program_to_instrument_name(i)
    for i in range(128)
}

# Build a 192-length boolean mask for continuous features
NORMALIZE_MASK = np.zeros(192, dtype=bool)

# basic continuous features
NORMALIZE_MASK[0] = True   # tempo
NORMALIZE_MASK[1] = True   # length
NORMALIZE_MASK[3] = True   # avg_num
NORMALIZE_MASK[4] = True   # avg_den

# rhythm (4)
NORMALIZE_MASK[157:161] = True

# harmony (19)
NORMALIZE_MASK[161:180] = True

# melody (4)
NORMALIZE_MASK[180:184] = True

# instrument strengths (7)
NORMALIZE_MASK[184:191] = True

# structure (1)
NORMALIZE_MASK[191] = True

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
    rhythm: np.ndarray | None = None
    harmony: np.ndarray | None = None
    melody: np.ndarray | None = None
    instr_strength: np.ndarray | None = None
    structure: np.ndarray | None = None

    def extract_feature_vector(self) -> np.ndarray:
        """
        Turns `MidiFeatures` into an `np.ndarray` that can be fed into a NN for training.

        Returns a vector of length:
            5 basic features +
            128 instrument one-hot +
            24 key one-hot +
            4 rhythm features +
            19 harmony features +
            4 melody features +
            7 instrument-family strength features +
            1 structure feature
            = 192 features total.

        A design choice here is to use separate values for major and minor of the same key, i.e.
        C major and C minor are represented differently as opposed to the same key. If our sample size is too small,
        (perhaps there are very few songs that have C minor), then we can configure this encoding such that 
        C major and C minor map to the same feature (so `key_vec = np.zeros(12), key_vec[k % 12] = 1.0`)

        The final vector is of form:
        [tempo, length, lyrics, avg_num, avg_den, *instrument_vec, *key_vec,
        *rhythm_vec, *harmony_vec, *melody_vec, *instrument_strength_vec, *structure_vec],
        where each element in the vector is of type `np.float64`
        """

        tempo = self.tempo
        length = self.length
        lyrics = 1.0 if self.lyrics else 0.0

        # Instruments
        # Convert list of ints representing instruments into a (one-hot) vector of length 128
        instrument_vec = np.zeros(128)
        for i in self.instruments:
            if 0 <= i < 128:
                instrument_vec[i] = 1.0

        # Key signatures
        # Convert list of ints representing key signatures into all possible keys
        key_vec = np.zeros(24)
        for k in self.key_signatures:
            if 0 <= k < 24:
                key_vec[k] = 1.0

        # Time signatures
        # encode as average numerator/denominator over the whole piece (don't know if this is the best thing to do
        # but I'd assume the time signatures don't change much over the whole piece?)
        if self.time_signatures:
            numerators, denominators = zip(*self.time_signatures)
            avg_num = np.mean(numerators)
            avg_den = np.mean(denominators)
        else:  # if there's no valid key signatures found from the midi file
            avg_num, avg_den = 4, 4  # default common time

        # Rhythm features: [note_density, spacing_variance, tempo_variance, syncopation_variance]
        rhythm_vec = self.rhythm if self.rhythm is not None else np.zeros(4)

        # Harmony features: [12 chroma + 6 tonnetz + chord_change_rate]
        harmony_vec = self.harmony if self.harmony is not None else np.zeros(19)

        # Melody features: [avg_pitch, pitch_var, interval_var, melodic_activity]
        melody_vec = self.melody if self.melody is not None else np.zeros(4)

        # Instrument family strength: [drums, strings, brass, woodwinds, piano, guitar, synth]
        instr_strength_vec = self.instr_strength if self.instr_strength is not None else np.zeros(7)

        # Structure: [structure_variance]
        structure_vec = self.structure if self.structure is not None else np.zeros(1)

        # combine vectors, return
        return np.concatenate([
            [tempo, length, lyrics, avg_num, avg_den],  # 5 basic features
            instrument_vec,                             # 128
            key_vec,                                     # 24
            rhythm_vec,                                  # 4
            harmony_vec,                                 # 19
            melody_vec,                                  # 4
            instr_strength_vec,                          # 7
            structure_vec,                               # 1
        ])


def compute_rhythm_features(midi: pretty_midi.PrettyMIDI, length: float) -> np.ndarray:
    notes = []
    for inst in midi.instruments:
        notes.extend(inst.notes)
    if len(notes) == 0:
        return np.zeros(4)

    onsets = np.array([n.start for n in notes])
    durations = np.array([n.end - n.start for n in notes])

    note_density = len(notes) / max(length, 1e-6)

    if len(onsets) > 2:
        spacing = np.diff(np.sort(onsets))
        density_var = np.var(spacing)
    else:
        density_var = 0.0

    iois = np.diff(np.sort(onsets))
    tempo_var = np.var(iois) if len(iois) > 1 else 0.0

    syncopation = np.var(durations) if len(durations) > 1 else 0.0

    return np.array([note_density, density_var, tempo_var, syncopation])

def compute_harmony_features(midi: pretty_midi.PrettyMIDI) -> np.ndarray:
    chroma = midi.get_chroma()
    avg_chroma = np.mean(chroma, axis=1)

    try:
        tonnetz = pretty_midi.utilities.get_tonnetz(midi)
        avg_tonnetz = np.mean(tonnetz, axis=1)
    except Exception:
        avg_tonnetz = np.zeros(6)

    diffs = np.diff(np.argmax(chroma, axis=0))
    chord_change_rate = np.mean(diffs != 0)

    return np.concatenate([avg_chroma, avg_tonnetz, [chord_change_rate]])

def compute_melody_features(midi: pretty_midi.PrettyMIDI, length: float) -> np.ndarray:
    pitches = []

    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            pitches.append(n.pitch)

    if len(pitches) == 0:
        return np.zeros(4)

    pitches = np.array(pitches)
    pitch_diffs = np.diff(np.sort(pitches))
    interval_var = np.var(pitch_diffs) if len(pitch_diffs) > 1 else 0.0

    return np.array([
        np.mean(pitches),
        np.var(pitches),
        interval_var,
        len(pitches) / max(length, 1e-6)
    ])

def compute_instrument_strength(midi: pretty_midi.PrettyMIDI) -> np.ndarray:
    family_counts = {
        "drums": 0,
        "strings": 0,
        "brass": 0,
        "woodwinds": 0,
        "piano": 0,
        "guitar": 0,
        "synth": 0,
    }

    total_notes = 0

    for inst in midi.instruments:
        count = len(inst.notes)
        total_notes += count
        name = PROGRAM_ID_TO_INSTRUMENT_NAME.get(inst.program, "").lower()

        if inst.is_drum:
            family_counts["drums"] += count
        elif "piano" in name:
            family_counts["piano"] += count
        elif "guitar" in name:
            family_counts["guitar"] += count
        elif "string" in name:
            family_counts["strings"] += count
        elif "brass" in name:
            family_counts["brass"] += count
        elif "wood" in name:
            family_counts["woodwinds"] += count
        elif "synth" in name:
            family_counts["synth"] += count

    if total_notes == 0:
        return np.zeros(7)

    return np.array([family_counts[k] / total_notes for k in family_counts.keys()])

def compute_structure_features(midi: pretty_midi.PrettyMIDI) -> np.ndarray:
    try:
        chroma = midi.get_chroma()
        num_cols = chroma.shape[1]
        chunks = 8
        step = max(num_cols // chunks, 1)

        segments = [np.mean(chroma[:, i:i+step], axis=1)
                    for i in range(0, num_cols, step)]

        sims = []
        for a, b in zip(segments[:-1], segments[1:]):
            sims.append(
                np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
            )

        structure_var = np.var(sims) if len(sims) > 1 else 0.0
        return np.array([structure_var])
    except Exception:
        return np.zeros(1)



@dataclass
class Track:
    """
    Contains information about a specific track
    """
    track_id: str
    genre: str
    features: MidiFeatures