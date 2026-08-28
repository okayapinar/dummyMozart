"""
AIRL (Adversarial Inverse Reinforcement Learning)

Sozde kod
----------
    Uzman gecislerini yukle: (s, a, s') ~ tau_E
    Ayrimci D_theta = {g, h} ve politika pi'yi baslat

    her AIRL iterasyonunda:
        Ajan gecislerini topla: (s, a, s') ~ pi

        her ayrimci epoch'unda:
            Uzman ve ajan batch'i ornekle
            Her iki batch icin log pi(a|s) hesapla
            f_theta(s, a, s') = g(s, a) + gamma * h(s') - h(s)
            logits = f_theta - log pi
            Loss = BCE(uzman, 1) + BCE(ajan, 0)
            Ayrimciyi guncelle

        Odul: r(s, a, s') = f_theta(s, a, s')
        Politikayi PPO ile bu odulu kullanarak guncelle
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


class AIRLDiscriminator(nn.Module):
    """Uzman vs ajan ayirimi yapan ag. Odul f_theta icinden turetilir."""

    def __init__(self, state_dim: int, action_dim: int = 1, hidden: int = 128, gamma: float = 0.99):
        super().__init__()
        self.gamma = gamma

        # g(s, a): durum-aksiyon oduulunun ogrenilen kismi
        self.g = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        # h(s): durum degeri; shaping icin kullanilir
        self.h = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def f_theta(self, state, action, next_state):
        # f_theta = g(s, a) + gamma * h(s') - h(s)
        action = action.view(-1, 1).float()
        state_action = torch.cat([state, action], dim=-1)
        return self.g(state_action) + self.gamma * self.h(next_state) - self.h(state)

    def compute_logits(self, state, action, next_state, log_pi):
        # Ayrimci skoru: ogrenilen odul eksi politikanin log olasiligi
        return self.f_theta(state, action, next_state) - log_pi.view(-1, 1)


class AIRLRewardWrapper(gym.Wrapper):
    """Ortamin ham odulunu AIRL'nin f_theta odulu ile degistirir."""

    def __init__(self, env, discriminator: AIRLDiscriminator):
        super().__init__(env)
        self.discriminator = discriminator
        self.current_state = None

    def reset(self, **kwargs):
        self.current_state, info = self.env.reset(**kwargs)
        return self.current_state, info

    def step(self, action):
        next_state, _, terminated, truncated, info = self.env.step(action)

        # PPO bu odulu kullanir; ayirimci egitilirken dondurulur
        s_t = torch.FloatTensor(self.current_state).unsqueeze(0)
        a_t = torch.tensor([action]).unsqueeze(0)
        s_next_t = torch.FloatTensor(next_state).unsqueeze(0)

        with torch.no_grad():
            airl_reward = self.discriminator.f_theta(s_t, a_t, s_next_t).item()

        self.current_state = next_state
        return next_state, airl_reward, terminated, truncated, info


def get_log_probs(policy, states, actions):
    # pi(a|s) log olasiligi; ayirimci logits'inde kullanilir
    states_t = torch.as_tensor(states, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.long)
    with torch.no_grad():
        return policy.log_prob(states_t, actions_t)


def collect_agent_trajectories(env, policy, n_steps=1000):
    # Mevcut politikadan (s, a, s') gecisleri topla
    states, actions, next_states = [], [], []
    state, _ = env.reset()

    for _ in range(n_steps):
        action, _ = policy.predict(state, deterministic=False)
        next_state, _, terminated, truncated, _ = env.step(action)

        states.append(state)
        actions.append(action)
        next_states.append(next_state)

        if terminated or truncated:
            state, _ = env.reset()
        else:
            state = next_state

    return np.asarray(states), np.asarray(actions), np.asarray(next_states)


def train_discriminator(discriminator, optimizer, expert_data, agent_data, policy, epochs=5, batch_size=64):
    # Uzmani 1, ajani 0 olarak etiketleyip ayirimciyi egit
    exp_s, exp_a, exp_next_s = expert_data
    agent_s, agent_a, agent_next_s = agent_data
    criterion = nn.BCEWithLogitsLoss()

    exp_batch = min(batch_size, len(exp_s))
    agent_batch = min(batch_size, len(agent_s))

    totals = {
        "disc/loss": 0.0,
        "disc/expert_acc": 0.0,
        "disc/agent_acc": 0.0,
        "disc/expert_reward": 0.0,
        "disc/agent_reward": 0.0,
    }

    for _ in range(epochs):
        # Her epoch'ta rastgele uzman ve ajan batch'i
        exp_idx = np.random.choice(len(exp_s), exp_batch, replace=len(exp_s) < exp_batch)
        agent_idx = np.random.choice(len(agent_s), agent_batch, replace=len(agent_s) < agent_batch)

        s_exp = torch.as_tensor(exp_s[exp_idx], dtype=torch.float32)
        a_exp = torch.as_tensor(exp_a[exp_idx], dtype=torch.long)
        s_next_exp = torch.as_tensor(exp_next_s[exp_idx], dtype=torch.float32)
        log_pi_exp = get_log_probs(policy, exp_s[exp_idx], exp_a[exp_idx])

        s_gen = torch.as_tensor(agent_s[agent_idx], dtype=torch.float32)
        a_gen = torch.as_tensor(agent_a[agent_idx], dtype=torch.long)
        s_next_gen = torch.as_tensor(agent_next_s[agent_idx], dtype=torch.float32)
        log_pi_gen = get_log_probs(policy, agent_s[agent_idx], agent_a[agent_idx])

        logits_exp = discriminator.compute_logits(s_exp, a_exp, s_next_exp, log_pi_exp)
        logits_gen = discriminator.compute_logits(s_gen, a_gen, s_next_gen, log_pi_gen)

        # Uzman "gercek", ajan "sahte"
        loss = criterion(logits_exp, torch.ones_like(logits_exp)) + criterion(logits_gen, torch.zeros_like(logits_gen))

        with torch.no_grad():
            totals["disc/loss"] += loss.item()
            totals["disc/expert_acc"] += (torch.sigmoid(logits_exp) > 0.5).float().mean().item()
            totals["disc/agent_acc"] += (torch.sigmoid(logits_gen) < 0.5).float().mean().item()
            totals["disc/expert_reward"] += discriminator.f_theta(s_exp, a_exp, s_next_exp).mean().item()
            totals["disc/agent_reward"] += discriminator.f_theta(s_gen, a_gen, s_next_gen).mean().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return {key: value / epochs for key, value in totals.items()}
