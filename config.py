from datetime import date
from pathlib import Path

from ml_collections import ConfigDict

from configs import load_config as _load_config

DEFAULT_CONFIG_NAME = "trial"
_active: ConfigDict | None = None


def load_config(name: str = DEFAULT_CONFIG_NAME) -> ConfigDict:
    return _load_config(name)


def set_active_config(cfg: ConfigDict) -> None:
    global _active
    _active = cfg


def get_active_config() -> ConfigDict:
    if _active is None:
        set_active_config(load_config())
    return _active


def min_sequence_length(cfg: ConfigDict | None = None) -> int:
    cfg = cfg or get_active_config()
    return cfg.data.seq_len + 1


def today_checkpoint_dir(cfg: ConfigDict | None = None) -> Path:
    cfg = cfg or get_active_config()
    return Path(cfg.paths.checkpoint_dir) / date.today().isoformat()


def today_checkpoint_path(cfg: ConfigDict | None = None) -> Path:
    cfg = cfg or get_active_config()
    return today_checkpoint_dir(cfg) / cfg.paths.checkpoint_name


def latest_checkpoint_path(cfg: ConfigDict | None = None) -> Path:
    cfg = cfg or get_active_config()
    checkpoint_dir = Path(cfg.paths.checkpoint_dir)
    checkpoint_name = cfg.paths.checkpoint_name
    matches = sorted(checkpoint_dir.glob(f"*/{checkpoint_name}"))
    if matches:
        return matches[-1]
    return today_checkpoint_path(cfg)
