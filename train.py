import argparse
import logging

import torch.optim as optim
from tqdm import tqdm

import config
from airl import (
    AIRLDiscriminator,
    AIRLRewardWrapper,
    collect_agent_trajectories,
    train_discriminator,
)
from dataset import build_expert_transitions
from env import MIDIMusicEnv
from midi_io import vocab_size
from ppo import PPOAgent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train AIRL music composer from MIDI files.")
    parser.add_argument("--iters", type=int, default=config.N_AIRL_ITERS)
    return parser.parse_args()


def main():
    args = parse_args()
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    expert_states, expert_actions, expert_next_states = build_expert_transitions()
    expert_data = (expert_states, expert_actions, expert_next_states)

    base_env = MIDIMusicEnv(
        expert_states=expert_states,
        midi_sequence_length=config.SEQ_LEN,
        vocab_size=vocab_size(),
        max_steps=config.ENV_MAX_STEPS,
    )

    discriminator = AIRLDiscriminator(state_dim=config.SEQ_LEN, action_dim=1)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=config.DISC_LR)
    airl_env = AIRLRewardWrapper(base_env, discriminator)
    generator = PPOAgent(airl_env)

    logger.info("Starting AIRL training with %d expert transitions.", len(expert_states))

    for _ in tqdm(range(args.iters), desc="AIRL"):
        agent_data = collect_agent_trajectories(base_env, generator.policy, n_steps=config.AGENT_COLLECT_STEPS)
        train_discriminator(
            discriminator=discriminator,
            optimizer=disc_optimizer,
            expert_data=expert_data,
            agent_data=agent_data,
            policy=generator.policy,
            epochs=config.DISC_EPOCHS,
            batch_size=config.BATCH_SIZE,
        )
        generator.learn(total_timesteps=config.PPO_TIMESTEPS)
        generator.save(str(config.CHECKPOINT_PATH))

    logger.info("Training complete. Checkpoint: %s", config.CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
