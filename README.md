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

Put expert MIDI files in `midis/` (subfolders are scanned). Files must be long enough to produce at least `seq_len + 1` tokens (64 + 1 by default).

## Training

```bash
python train.py
python train.py --config prod
python train.py --config debug
python train.py --config trial --iters 50
python train.py --logdir runs/experiment1
```

The default config is `prod` (`configs/prod.py`). Each round:

- the agent collects `airl.agent_collect_steps` steps
- the discriminator is trained for `airl.disc_epochs` epochs
- PPO is updated for `ppo.timesteps` steps
- a checkpoint is saved under `checkpoints/<date>/ppo.pt`
- scalars are written to `runs/<timestamp>/` (or `--logdir`)

Watch training with TensorBoard:

```bash
tensorboard --logdir runs
```

Then open http://localhost:6006. Scalars are grouped as `disc/*` (loss, expert/agent accuracy and reward) and `ppo/*` (policy/value loss, entropy, clip fraction, KL, reward, explained variance).

## Composing

```bash
python compose.py
python compose.py --config prod
python compose.py --steps 256
python compose.py --seed-midi midis/example.mid --output output/piece.mid
```

| Argument | Description |
|---|---|
| `--config` | Config preset: `prod`, `debug`, or `trial` (default: `prod`) |
| `--checkpoint` | PPO weights (default: latest under `checkpoints/`) |
| `--steps` | Number of tokens to generate (default: 128) |
| `--seed-midi` | MIDI used as context (at least 64 tokens) |
| `--output` | Output path (otherwise `output/compose_<time>.mid`) |

## Files

| File | Role |
|---|---|
| `train.py` | AIRL loop and TensorBoard logging |
| `compose.py` | MIDI generation from a checkpoint |
| `config.py` | Config loader and path helpers |
| `configs/` | ml_collections presets (`prod`, `debug`, `trial`) |
| `dataset.py` | MIDI → expert transitions |
| `midi_io.py` | REMI encode / decode |
| `env.py` | Gymnasium sliding-window environment |
| `airl.py` | Discriminator, reward wrapper, trajectory collection |
| `ppo.py` | Actor-critic, GAE, clipped PPO |

## Settings

Configs use `ml_collections.ConfigDict` under `configs/`:

| Preset | Purpose |
|---|---|
| `prod` | Full training (default; matches original hyperparameters) |
| `debug` | Fast smoke test (few iters, small batches) |
| `trial` | Medium experiment run (smaller model, more exploration) |

Shared defaults live in `configs/base.py`. Key groups:

- `paths` — `midi_dir`, `output_dir`, `checkpoint_dir`, `log_dir`
- `data` — `seq_len` (64), `country_filter`, `tokenizer`
- `airl` — `n_iters`, `disc_lr`, `batch_size`, ...
- `ppo` — `lr`, `clip`, `ent_coef`, ...

Country filter: set `data.country_filter = "England"` in a preset to use only `midis/England/`.

## Requirements

- gymnasium
- torch
- numpy
- miditok
- tqdm
- tensorboard
- ml_collections
- stable-baselines3 (listed at install time; training uses the project's own PPO)
