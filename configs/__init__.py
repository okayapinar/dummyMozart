from ml_collections import ConfigDict

from configs import debug, prod, trial

_CONFIGS = {
    "prod": prod.get_config,
    "debug": debug.get_config,
    "trial": trial.get_config,
}


def load_config(name: str = "prod") -> ConfigDict:
    if name not in _CONFIGS:
        raise ValueError(f"Unknown config: {name}. Choose from {list(_CONFIGS)}")
    return _CONFIGS[name]()


def list_configs() -> list[str]:
    return list(_CONFIGS)
