from dataclasses import dataclass

import torch


@dataclass
class Transition:
    state: object
    action: int
    reward: float
    done: bool
    log_prob: float
    value: float


@dataclass
class Rollout:
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
