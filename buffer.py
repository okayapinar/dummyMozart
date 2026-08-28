import torch
from torch._dynamo import callback

from datatypes import Rollout, Transition


def compute_gae(rewards, values, dones, last_value, gamma: float, gae_lambda: float):
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    next_value = last_value
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae
        next_value = values[t]
    return advantages, advantages + values


class RolloutBuffer:
    def __init__(self, capacity: int, state_dim: int, gamma: float, gae_lambda: float):
        self.capacity = capacity
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ptr = 0

        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.dones = torch.zeros(capacity, dtype=torch.float32)
        self.log_probs = torch.zeros(capacity, dtype=torch.float32)
        self.values = torch.zeros(capacity, dtype=torch.float32)
        self.advantages = None
        self.returns = None

    def reset(self) -> None:
        self.ptr = 0
        self.advantages = None
        self.returns = None

    def add(self, transition: Transition) -> None:
        if self.ptr >= self.capacity:
            raise IndexError("RolloutBuffer is full")
        self.states[self.ptr] = torch.as_tensor(transition.state, dtype=torch.float32)
        self.actions[self.ptr] = transition.action
        self.rewards[self.ptr] = transition.reward
        self.dones[self.ptr] = float(transition.done)
        self.log_probs[self.ptr] = transition.log_prob
        self.values[self.ptr] = transition.value
        self.ptr += 1

    def finalize(self, last_value: float) -> None:
        rewards = self.rewards[: self.ptr]
        values = self.values[: self.ptr]
        dones = self.dones[: self.ptr]
        self.advantages, self.returns = compute_gae(rewards, values, dones, last_value, self.gamma, self.gae_lambda)

    def get(self) -> Rollout:
        n = self.ptr
        return Rollout(
            states=self.states[:n],
            actions=self.actions[:n],
            log_probs=self.log_probs[:n],
            advantages=self.advantages,
            returns=self.returns,
        )

    def __len__(self) -> int:
        return self.ptr