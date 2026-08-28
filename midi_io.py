from pathlib import Path

from miditok import REMI, TokenizerConfig
from miditok.classes import TokSequence
from ml_collections import ConfigDict

import config


def _tokenizer_params(cfg: ConfigDict) -> dict:
    return cfg.data.tokenizer.to_dict()


def get_tokenizer(cfg: ConfigDict | None = None) -> REMI:
    cfg = cfg or config.get_active_config()
    return REMI(TokenizerConfig(**_tokenizer_params(cfg)))


def vocab_size(cfg: ConfigDict | None = None) -> int:
    return len(get_tokenizer(cfg))


def load_token_sequence(path: Path | str, cfg: ConfigDict | None = None) -> list[int]:
    tokens = get_tokenizer(cfg)(Path(path))
    if not isinstance(tokens, list):
        return tokens.ids
    if not tokens:
        return []
    longest = tokens[0]
    for seq in tokens[1:]:
        if len(seq.ids) > len(longest.ids):
            longest = seq
    return longest.ids


def tokens_to_midi(token_ids: list[int], out_path: Path | str, cfg: ConfigDict | None = None) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    score = get_tokenizer(cfg).decode([TokSequence(ids=list(token_ids))])
    score.dump_midi(out_path)
