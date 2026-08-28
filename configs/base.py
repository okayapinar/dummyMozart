from pathlib import Path

from ml_collections import ConfigDict


def get_base_config() -> ConfigDict:
    root = Path(__file__).resolve().parent.parent

    cfg = ConfigDict()
    cfg.paths = ConfigDict()
    cfg.paths.root = root
    cfg.paths.midi_dir = root / "midis"
    cfg.paths.output_dir = root / "output"
    cfg.paths.checkpoint_dir = root / "checkpoints"
    cfg.paths.checkpoint_name = "ppo.pt"
    cfg.paths.log_dir = root / "runs"

    cfg.data = ConfigDict()
    cfg.data.seq_len = 64
    cfg.data.country_filter = None
    cfg.data.tokenizer = ConfigDict()
    cfg.data.tokenizer.pitch_range = (21, 109)
    cfg.data.tokenizer.beat_res = {(0, 4): 8, (4, 12): 4}
    cfg.data.tokenizer.num_velocities = 16
    cfg.data.tokenizer.use_chords = False
    cfg.data.tokenizer.use_rests = False
    cfg.data.tokenizer.use_tempos = False
    cfg.data.tokenizer.use_time_signatures = False
    cfg.data.tokenizer.use_programs = False

    cfg.env = ConfigDict()
    cfg.env.max_steps = 100

    cfg.airl = ConfigDict()
    cfg.airl.n_iters = 100
    cfg.airl.agent_collect_steps = 1024
    cfg.airl.disc_epochs = 5
    cfg.airl.disc_lr = 3e-4
    cfg.airl.batch_size = 256

    cfg.ppo = ConfigDict()
    cfg.ppo.hidden = 128
    cfg.ppo.timesteps = 2048
    cfg.ppo.lr = 3e-4
    cfg.ppo.gamma = 0.99
    cfg.ppo.gae_lambda = 0.95
    cfg.ppo.clip = 0.2
    cfg.ppo.epochs = 10
    cfg.ppo.minibatch = 256
    cfg.ppo.ent_coef = 0.02
    cfg.ppo.vf_coef = 0.5
    cfg.ppo.max_grad_norm = 0.5

    return cfg
