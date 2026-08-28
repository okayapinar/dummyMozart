# dummyMozart

An AIRL + PPO project that learns expert behavior from MIDI files and generates new music.

Instead of designing a reward by hand, transitions are extracted from pieces under `midis/`. The discriminator (AIRL) tells expert from agent; PPO uses that reward to learn the next token. The trained policy writes a new MIDI file.

## How it works

1. MIDI files are converted to token sequences with a REMI tokenizer.
2. Each window is an expert transition `(state, action, next state)`: the state is the last 64 tokens, the action is the next token.
3. The environment (`MIDIMusicEnv`) is a sliding window; the action is appended to the end of the window.
4. Each AIRL round collects agent trajectories, trains the discriminator, and updates PPO with the `f_θ` reward.
5. `compose.py` generates tokens from a checkpoint and converts them to MIDI.

```
midis/*.mid  →  expert transitions  →  AIRL (discriminator + PPO)  →  checkpoints/ppo.pt  →  output/*.mid
```

## AIRL pseudocode

Training follows this loop in `airl.py`. The discriminator learns to tell expert from agent; PPO updates the policy with the discriminator's reward.

```
Load expert transitions: (s, a, s') ~ τ_E
Initialize discriminator D_θ = {g, h} and policy π

for each AIRL iteration:
    Collect agent transitions: (s, a, s') ~ π

    for each discriminator epoch:
        Sample an expert batch and an agent batch
        Compute log π(a|s) for both batches
        f_θ(s, a, s') = g(s, a) + γ · h(s') − h(s)
        logits = f_θ − log π
        Loss = BCE(expert, 1) + BCE(agent, 0)
        Update the discriminator

    Reward: r(s, a, s') = f_θ(s, a, s')
    Update the policy with PPO using this reward
```

- `g(s, a)` is the learned state-action reward; `h(s)` is the potential (shaping) function.
- `f_θ` is the shaped reward; the `h` terms make learning easier without changing the policy optimum.
- Subtracting `log π` from the logits lets the discriminator ask “is this transition expert?” independently of the policy.
- Expert is labeled 1, agent 0 (BCE). PPO uses `f_θ` as the environment reward.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Put expert MIDI files in `midis/` (subfolders are scanned). Files must be long enough to produce at least `SEQ_LEN + 1` tokens.

## Training

```bash
python train.py
python train.py --iters 50
```

The default iteration count is `config.N_AIRL_ITERS` (100). Each round:

- the agent collects `AGENT_COLLECT_STEPS` steps
- the discriminator is trained for `DISC_EPOCHS` epochs
- PPO is updated for `PPO_TIMESTEPS` steps
- a checkpoint is saved as `checkpoints/ppo.pt`

## Composing

```bash
python compose.py
python compose.py --steps 256
python compose.py --seed-midi midis/example.mid --output output/piece.mid
```

| Argument | Description |
|---|---|
| `--checkpoint` | PPO weights (default: `checkpoints/ppo.pt`) |
| `--steps` | Number of tokens to generate (default: 128) |
| `--seed-midi` | MIDI used as context (at least 64 tokens) |
| `--output` | Output path (otherwise `output/compose_<time>.mid`) |

## Files

| File | Role |
|---|---|
| `train.py` | AIRL loop |
| `compose.py` | MIDI generation from a checkpoint |
| `config.py` | Paths, tokenizer, AIRL and PPO hyperparameters |
| `dataset.py` | MIDI → expert transitions |
| `midi_io.py` | REMI encode / decode |
| `env.py` | Gymnasium sliding-window environment |
| `airl.py` | Discriminator, reward wrapper, trajectory collection |
| `ppo.py` | Actor-critic, GAE, clipped PPO |

## Settings

Important values live in `config.py`:

- `MIDI_DIR`, `SEQ_LEN` (64), `COUNTRY_FILTER` — data
- `N_AIRL_ITERS`, `DISC_LR`, `BATCH_SIZE` — AIRL
- `PPO_LR`, `PPO_CLIP`, `PPO_ENT_COEF` — policy

Country filter: `COUNTRY_FILTER = "England"` uses only `midis/England/`.

## Requirements

- gymnasium
- torch
- numpy
- miditok
- tqdm
- stable-baselines3 (listed at install time; training uses the project's own PPO)
