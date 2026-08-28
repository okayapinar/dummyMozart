import logging
from pathlib import Path

import numpy as np

import config
from midi_io import load_pitch_sequence

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
            pitches = load_pitch_sequence(midi_path)
        except Exception as exc:
            logger.warning("Skipping %s: %s", midi_path, exc)
            skipped += 1
            continue

        if len(pitches) < seq_len + 1:
            skipped += 1
            continue

        pitch_array = np.array(pitches, dtype=np.float32)
        for i in range(len(pitch_array) - seq_len):
            state = pitch_array[i : i + seq_len]
            action = int(pitch_array[i + seq_len])
            next_state = pitch_array[i + 1 : i + seq_len + 1]
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


def save_expert_cache(
    cache_path: Path | str,
    expert_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    states, actions, next_states = expert_data
    np.savez_compressed(
        cache_path,
        states=states,
        actions=actions,
        next_states=next_states,
    )


def load_expert_cache(cache_path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_path = Path(cache_path)
    data = np.load(cache_path)
    return data["states"], data["actions"], data["next_states"]


def load_or_build_expert(
    cache_path: Path | str = config.EXPERT_CACHE,
    midi_dir: Path | str = config.MIDI_DIR,
    seq_len: int = config.SEQ_LEN,
    country_filter: str | None = config.COUNTRY_FILTER,
    rebuild: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_path = Path(cache_path)

    if cache_path.exists() and not rebuild:
        logger.info("Loading expert cache from %s", cache_path)
        return load_expert_cache(cache_path)

    logger.info("Building expert transitions from %s", midi_dir)
    expert_data = build_expert_transitions(midi_dir, seq_len, country_filter)
    save_expert_cache(cache_path, expert_data)
    return expert_data
