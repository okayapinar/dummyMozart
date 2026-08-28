import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MIDIMusicEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        expert_states: np.ndarray,
        midi_sequence_length: int = 16,
        vocab_size: int = 128,
        max_steps: int = 100,
    ):
        super().__init__()

        if len(expert_states) == 0:
            raise ValueError("expert_states must not be empty.")

        self.sequence_length = midi_sequence_length
        self.vocab_size = vocab_size
        self.max_steps = max_steps
        self.expert_states = expert_states.astype(np.float32)

        self.observation_space = spaces.Box(
            low=0,
            high=self.vocab_size - 1,
            shape=(self.sequence_length,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.vocab_size)

        self.current_step = 0
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        idx = self.np_random.integers(0, len(self.expert_states))
        self.state = self.expert_states[idx].copy()
        return self.state, {}

    def step(self, action):
        self.current_step += 1

        new_state = np.roll(self.state, shift=-1)
        new_state[-1] = float(action)
        self.state = new_state

        terminated = self.current_step >= self.max_steps
        truncated = False
        return self.state, 0.0, terminated, truncated, {}

    def render(self):
        print(f"Step: {self.current_step}, Last notes: {self.state[-4:]}")
