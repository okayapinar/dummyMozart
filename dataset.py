import logging
from pathlib import Path

import numpy as np

import config
from midi_io import load_token_sequence

logger = logging.getLogger(__name__)


def _iter_midi_files(midi_dir: Path, country_filter: str | None = None) -> list[Path]:
    if country_filter:
        search_root = midi_dir / country_filter
        if not search_root.exists():
            raise FileNotFoundError(f"Country folder not found: {search_root}")
        return sorted(search_root.rglob("*.mid"))

    return sorted(midi_dir.rglob("*.mid"))


def build_expert_transitions(
    midi_dir: Path | str = config.MIDI_DIR, seq_len: int = config.SEQ_LEN, country_filter: str | None = config.COUNTRY_FILTER
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (state, action, next_state) tuples from all MIDI files."""
    midi_dir = Path(midi_dir)
    midi_files = _iter_midi_files(midi_dir, country_filter)
    if not midi_files:
        raise RuntimeError(f"No MIDI files found under {midi_dir}")

    states: list[np.ndarray] = []
    actions: list[int] = []
    next_states: list[np.ndarray] = []
    skipped = 0

    for midi_path in midi_files:
        try:
            token_ids = load_token_sequence(midi_path)
        except Exception as exc:
            logger.warning("Skipping %s: %s", midi_path, exc)
            skipped += 1
            continue

        if len(token_ids) < seq_len + 1:
            skipped += 1
            continue

        token_array = np.array(token_ids, dtype=np.float32)
        for i in range(len(token_array) - seq_len):
            state = token_array[i : i + seq_len]
            action = int(token_array[i + seq_len])
            next_state = token_array[i + 1 : i + seq_len + 1]
            states.append(state)
            actions.append(action)
            next_states.append(next_state)

    if not states:
        raise RuntimeError("No expert transitions could be built from MIDI files.")

    logger.info("Built %d transitions from %d files (%d skipped).", len(states), len(midi_files), skipped)
    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(actions, dtype=np.int64),
        np.asarray(next_states, dtype=np.float32),
    )
