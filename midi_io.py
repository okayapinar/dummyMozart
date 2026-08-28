from pathlib import Path

from miditok import REMI, TokenizerConfig
from miditok.classes import TokSequence

import config


def get_tokenizer() -> REMI:
    return REMI(TokenizerConfig(**config.TOKENIZER_PARAMS))


def vocab_size() -> int:
    return len(get_tokenizer())


def load_token_sequence(path: Path | str) -> list[int]:
    tokens = get_tokenizer()(Path(path))
    if not isinstance(tokens, list):
        return tokens.ids
    if not tokens:
        return []
    longest = tokens[0]
    for seq in tokens[1:]:
        if len(seq.ids) > len(longest.ids):
            longest = seq
    return longest.ids


def tokens_to_midi(token_ids: list[int], out_path: Path | str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    score = get_tokenizer().decode([TokSequence(ids=list(token_ids))])
    score.dump_midi(out_path)
