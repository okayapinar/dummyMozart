from ml_collections import ConfigDict

from configs.base import get_base_config


def get_config() -> ConfigDict:
    cfg = get_base_config()
    cfg.airl.n_iters = 3
    cfg.airl.agent_collect_steps = 128
    cfg.airl.disc_epochs = 2
    cfg.airl.batch_size = 64
    cfg.ppo.timesteps = 256
    cfg.ppo.epochs = 2
    cfg.ppo.minibatch = 64
    return cfg
