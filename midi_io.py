import logging
from pathlib import Path

import numpy as np
import pretty_midi

import config

logger = logging.getLogger(__name__)


def _collect_notes(midi_data: pretty_midi.PrettyMIDI) -> list[pretty_midi.Note]:
    if midi_data.instruments:
        return list(midi_data.instruments[0].notes)
    notes: list[pretty_midi.Note] = []
    for instrument in midi_data.instruments:
        notes.extend(instrument.notes)
    return notes


def load_pitch_sequence(path: Path | str, quantize_sec: float = config.QUANTIZE_SEC) -> list[int]:
    """Extract a monophonic pitch sequence from a MIDI file."""
    midi_data = pretty_midi.PrettyMIDI(str(path))
    notes = _collect_notes(midi_data)
    if not notes:
        return []

    end_time = max(note.end for note in notes)
    if end_time <= 0:
        return []

    grid_size = max(1, int(np.ceil(end_time / quantize_sec)))
    grid = np.full(grid_size, -1, dtype=np.int16)

    for note in notes:
        start_idx = int(note.start / quantize_sec)
        end_idx = min(grid_size, int(np.ceil(note.end / quantize_sec)))
        for idx in range(start_idx, end_idx):
            if idx < 0:
                continue
            if grid[idx] == -1 or note.pitch > grid[idx]:
                grid[idx] = note.pitch

    pitches: list[int] = []
    for value in grid:
        if value == -1:
            continue
        pitch = int(np.clip(value, 0, config.VOCAB_SIZE - 1))
        pitches.append(pitch)

    return pitches


def sequence_to_midi(
    pitches: list[int] | np.ndarray,
    out_path: Path | str,
    quantize_sec: float = config.QUANTIZE_SEC,
    velocity: int = 90,
) -> None:
    """Write a monophonic pitch sequence to a MIDI file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    midi_data = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    start = 0.0

    for pitch in pitches:
        pitch = int(np.clip(pitch, 0, config.VOCAB_SIZE - 1))
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start,
            end=start + quantize_sec * 0.95,
        )
        instrument.notes.append(note)
        start += quantize_sec

    midi_data.instruments.append(instrument)
    midi_data.write(str(out_path))
