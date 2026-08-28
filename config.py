from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Paths
MIDI_DIR = ROOT / "midis"
OUTPUT_DIR = ROOT / "output"
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "ppo.pt"

# Data
SEQ_LEN = 64
MIN_SEQUENCE_LENGTH = SEQ_LEN + 1
COUNTRY_FILTER = None  # e.g. "England" to filter by country folder

TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 4): 8, (4, 12): 4},
    "num_velocities": 16,
    "use_chords": False,
    "use_rests": False,
    "use_tempos": False,
    "use_time_signatures": False,
    "use_programs": False,
}

# Environment
ENV_MAX_STEPS = 100

# AIRL
N_AIRL_ITERS = 100
AGENT_COLLECT_STEPS = 1024
DISC_EPOCHS = 5
DISC_LR = 3e-4
BATCH_SIZE = 64

# PPO
PPO_HIDDEN = 128
PPO_TIMESTEPS = 2048
PPO_LR = 3e-4
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP = 0.2
PPO_EPOCHS = 10
PPO_MINIBATCH = 64
PPO_ENT_COEF = 0.02
PPO_VF_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5
