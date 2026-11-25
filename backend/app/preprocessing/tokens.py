"""Offline WordPiece budgets using the pinned MiniLM tokenizer (not an LLM)."""

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from tokenizers import Tokenizer  # type: ignore[import-untyped]

TOKENIZER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
TOKENIZER_SHA256 = "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"
TEXT_PROFILE = "minilm-wordpiece-v2"


@lru_cache(maxsize=1)
def tokenizer() -> Tokenizer:
    path = Path(__file__).parent / "assets" / "minilm_tokenizer.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("Pinned tokenizer asset checksum mismatch")
    instance = Tokenizer.from_file(str(path))
    instance.no_truncation()
    instance.no_padding()
    return instance


def token_count(text: str) -> int:
    """Include CLS/SEP special tokens in every count."""
    return len(tokenizer().encode(text).ids)


@dataclass(frozen=True)
class BudgetedText:
    text: str
    original_tokens: int
    tokens: int
    truncated: bool


def token_prefix(text: str, budget: int) -> BudgetedText:
    instance = tokenizer()
    original = len(instance.encode(text).ids)
    if original <= budget:
        return BudgetedText(text, original, original, False)
    capacity = budget - instance.num_special_tokens_to_add(False)
    if capacity < 1:
        raise ValueError("Token budget must leave room for content and special tokens")
    offsets = instance.encode(text, add_special_tokens=False).offsets
    while capacity:
        prefix = text[: offsets[capacity - 1][1]].rstrip()
        count = token_count(prefix)
        if count <= budget:
            return BudgetedText(prefix, original, count, True)
        capacity -= 1
    raise ValueError("Could not fit text within token budget")
