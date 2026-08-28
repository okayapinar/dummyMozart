from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.state_dim = int(state_dim)
        self.n_actions = int(n_actions)
        self.hidden = int(hidden)

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def _dist(self, states: torch.Tensor) -> Categorical:
        return Categorical(logits=self.actor(states))

    def act(self, state) -> tuple[int, float, float]:
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        dist = self._dist(state_t)
        action = dist.sample()
        value = self.critic(state_t)
        return int(action.item()), float(dist.log_prob(action).item()), float(value.item())

    def predict(self, state, deterministic: bool = False) -> tuple[int, None]:
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            dist = self._dist(state_t)
            action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        return int(action.item()), None

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self._dist(states).log_prob(actions)

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        dist = self._dist(states)
        values = self.critic(states).squeeze(-1)
        return dist.log_prob(actions), values, dist.entropy()


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


@dataclass
class Transition:
    state: object
    action: int
    reward: float
    done: bool
    log_prob: float
    value: float


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

    def get(self) -> dict[str, torch.Tensor]:
        n = self.ptr
        return {
            "states": self.states[:n],
            "actions": self.actions[:n],
            "log_probs": self.log_probs[:n],
            "advantages": self.advantages,
            "returns": self.returns,
        }

    def __len__(self) -> int:
        return self.ptr


class PPOAgent:
    def __init__(
        self,
        env,
        hidden: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        n_epochs: int = 10,
        minibatch_size: int = 64,
        max_grad_norm: float = 0.5,
        n_steps: int = 2048,
    ):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps

        state_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n
        self.policy = ActorCritic(state_dim, n_actions, hidden)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = RolloutBuffer(
            capacity=n_steps,
            state_dim=state_dim,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        self._last_state = None

    def collect_rollout(self, n_steps: int) -> dict[str, torch.Tensor]:
        self.buffer.reset()

        if self._last_state is None:
            self._last_state, _ = self.env.reset()
        state = self._last_state

        for _ in range(n_steps):
            with torch.no_grad():
                action, log_prob, value = self.policy.act(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            transition = Transition(
                state=state,
                action=action,
                reward=reward,
                done=done,
                log_prob=log_prob,
                value=value,
            )

            self.buffer.add(transition)
            state, _ = self.env.reset() if done else (next_state, None)

        self._last_state = state
        with torch.no_grad():
            last_value = self.policy.critic(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)).item()

        self.buffer.finalize(last_value)
        return self.buffer.get()

    def update(self, rollout: dict[str, torch.Tensor]) -> None:
        advantages = rollout["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(advantages)
        for _ in range(self.n_epochs):
            order = torch.randperm(n)
            for start in range(0, n, self.minibatch_size):
                idx = order[start : start + self.minibatch_size]
                new_log_probs, values, entropy = self.policy.evaluate(rollout["states"][idx], rollout["actions"][idx])
                ratio = torch.exp(new_log_probs - rollout["log_probs"][idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (values - rollout["returns"][idx]).pow(2).mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def predict(self, state, deterministic: bool = False) -> tuple[int, None]:
        return self.policy.predict(state, deterministic=deterministic)

    def learn(self, total_timesteps: int) -> None:
        steps_done = 0
        while steps_done < total_timesteps:
            n_steps = min(self.n_steps, total_timesteps - steps_done)
            self.update(self.collect_rollout(n_steps))
            steps_done += n_steps

    def save(self, path: str) -> None:
        torch.save(self.policy.state_dict(), path)

    @classmethod
    def load(cls, path: str, env) -> "PPOAgent":
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        if "policy" in state_dict:
            state_dict = state_dict["policy"]
        hidden = state_dict["actor.0.weight"].shape[0]
        agent = cls(env, hidden=hidden)
        agent.policy.load_state_dict(state_dict)
        agent.policy.eval()
        return agent
