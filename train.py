import argparse
import logging
from datetime import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm import tqdm

import config
from configs import list_configs
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
    parser.add_argument("--config", type=str, default=config.DEFAULT_CONFIG_NAME)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = config.load_config(args.config)
    config.set_active_config(cfg)

    n_iters = cfg.airl.n_iters
    checkpoint_dir = config.today_checkpoint_dir(cfg)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / cfg.paths.checkpoint_name

    log_dir = Path(cfg.paths.log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")

    expert_states, expert_actions, expert_next_states = build_expert_transitions(cfg=cfg)
    expert_data = (expert_states, expert_actions, expert_next_states)

    base_env = MIDIMusicEnv(
        expert_states=expert_states,
        midi_sequence_length=cfg.data.seq_len,
        vocab_size=vocab_size(cfg=cfg),
        max_steps=cfg.env.max_steps,
    )

    discriminator = AIRLDiscriminator(state_dim=cfg.data.seq_len, action_dim=1, gamma=cfg.ppo.gamma)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=cfg.airl.disc_lr)
    airl_env = AIRLRewardWrapper(base_env, discriminator)
    generator = PPOAgent(
        airl_env,
        hidden=cfg.ppo.hidden,
        lr=cfg.ppo.lr,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        clip_coef=cfg.ppo.clip,
        ent_coef=cfg.ppo.ent_coef,
        vf_coef=cfg.ppo.vf_coef,
        n_epochs=cfg.ppo.epochs,
        minibatch_size=cfg.ppo.minibatch,
        max_grad_norm=cfg.ppo.max_grad_norm,
        n_steps=cfg.ppo.timesteps,
    )

    logger.info("Config: %s", args.config)
    logger.info("Starting AIRL training with %d expert transitions.", len(expert_states))

    writer = SummaryWriter(str(log_dir))
    logger.info("TensorBoard log dir: %s", log_dir)
    try:
        for iteration in tqdm(range(n_iters), desc="AIRL"):
            agent_data = collect_agent_trajectories(base_env, generator.policy, n_steps=cfg.airl.agent_collect_steps)
            disc_metrics = train_discriminator(
                discriminator=discriminator,
                optimizer=disc_optimizer,
                expert_data=expert_data,
                agent_data=agent_data,
                policy=generator.policy,
                epochs=cfg.airl.disc_epochs,
                batch_size=cfg.airl.batch_size,
            )
            ppo_metrics = generator.learn(total_timesteps=cfg.ppo.timesteps)
            # generator.save(str(checkpoint_path))

            for key, value in {**disc_metrics, **ppo_metrics}.items():
                writer.add_scalar(key, value, iteration)
            writer.flush()
    finally:
        writer.close()

    logger.info("Training complete. Checkpoint: %s", checkpoint_path)
    logger.info("TensorBoard: tensorboard --logdir %s", cfg.paths.log_dir)


if __name__ == "__main__":
    main()
