from ml_collections import ConfigDict

from configs.base import get_base_config


def get_config() -> ConfigDict:
    cfg = get_base_config()
    cfg.airl.n_iters = 20
    cfg.airl.agent_collect_steps = 512
    cfg.ppo.timesteps = 1024
    cfg.ppo.hidden = 64
    cfg.ppo.ent_coef = 0.05
    return cfg
