import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

import config
from configs import list_configs
from dataset import build_expert_transitions
from env import MIDIMusicEnv
from midi_io import load_token_sequence, tokens_to_midi, vocab_size
from ppo import PPOAgent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Compose MIDI music from a trained PPO checkpoint.")
    parser.add_argument(
        "--config",
        type=str,
        default=config.DEFAULT_CONFIG_NAME,
        choices=list_configs(),
        help="Config preset (default: prod).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to PPO checkpoint (.pt).",
    )
    parser.add_argument("--steps", type=int, default=128, help="Number of tokens to generate.")
    parser.add_argument(
        "--seed-midi",
        type=str,
        default=None,
        help="Optional MIDI file to use as seed context.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output MIDI path (default: output/compose_<timestamp>.mid).",
    )
    return parser.parse_args()


def _seed_state_from_midi(seed_midi: Path, seq_len: int, cfg) -> np.ndarray:
    token_ids = load_token_sequence(seed_midi, cfg=cfg)
    if len(token_ids) < seq_len:
        raise ValueError(f"Seed MIDI too short (need at least {seq_len} tokens).")
    return np.array(token_ids[:seq_len], dtype=np.float32)


def compose(checkpoint: Path, steps: int, cfg, seed_state: np.ndarray | None = None) -> list[int]:
    expert_states, _, _ = build_expert_transitions(cfg=cfg)
    env = MIDIMusicEnv(
        expert_states=expert_states,
        midi_sequence_length=cfg.data.seq_len,
        vocab_size=vocab_size(cfg=cfg),
        max_steps=steps,
    )

    model = PPOAgent.load(str(checkpoint), env=env)

    if seed_state is None:
        state, _ = env.reset()
    else:
        env.state = seed_state.copy()
        env.current_step = 0
        state = env.state

    token_ids = list(state.astype(int))

    for _ in range(steps):
        action, _ = model.predict(state, deterministic=True)
        action = int(action)
        state, _, terminated, truncated, _ = env.step(action)
        token_ids.append(action)
        if terminated or truncated:
            break

    return token_ids


def main():
    args = parse_args()
    cfg = config.load_config(args.config)
    config.set_active_config(cfg)

    checkpoint = Path(args.checkpoint) if args.checkpoint else config.latest_checkpoint_path(cfg)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    seed_state = None
    if args.seed_midi:
        seed_state = _seed_state_from_midi(Path(args.seed_midi), cfg.data.seq_len, cfg)

    token_ids = compose(checkpoint, args.steps, cfg, seed_state)

    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(cfg.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"compose_{timestamp}.mid"

    tokens_to_midi(token_ids, output_path, cfg=cfg)
    logger.info("Saved composition (%d tokens) to %s", len(token_ids), output_path)


if __name__ == "__main__":
    main()
