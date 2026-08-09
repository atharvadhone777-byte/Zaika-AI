"""
Ingredient vocabulary: maps normalized ingredient strings <-> integer ids.

This is intentionally a flat ingredient-level vocabulary, not a subword/BPE
tokenizer. Rationale: recipe ingredients are closer to a closed categorical
vocabulary (~2-3k distinct normalized ingredients even at 180K recipes)
than to open-ended natural language, so subword tokenization would add
complexity (merge tables, unknown-piece handling) without adding value -
splitting "tomato" into subword pieces doesn't help a retrieval model that
only ever needs to know "was tomato present, yes/no" and how it co-occurs
with other whole ingredients. Full text preprocessing (subwords/wordpiece)
is used instead in rag/ for the free-text assistant, where language is
genuinely open-ended - a good example of choosing tokenization strategy
per-task rather than defaulting to one approach everywhere.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class IngredientVocabulary:
    def __init__(self, token_to_id: dict[str, int] | None = None):
        self.token_to_id: dict[str, int] = token_to_id or {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.id_to_token: dict[int, str] = {i: t for t, i in self.token_to_id.items()}

    @classmethod
    def build(cls, ingredient_lists: list[list[str]], min_frequency: int) -> "IngredientVocabulary":
        counts = Counter(ing for ings in ingredient_lists for ing in ings)
        vocab = cls()
        # Sorted for determinism: dict/Counter iteration order in Python is
        # insertion-order, which would make vocab ids depend on row order in
        # the (possibly re-shuffled) dataframe. Sorting by (frequency desc,
        # name) makes vocab-building reproducible independent of row order.
        kept = sorted(
            (tok for tok, c in counts.items() if c >= min_frequency),
            key=lambda t: (-counts[t], t),
        )
        for tok in kept:
            idx = len(vocab.token_to_id)
            vocab.token_to_id[tok] = idx
            vocab.id_to_token[idx] = tok
        return vocab

    def encode(self, ingredients: list[str]) -> list[int]:
        unk_id = self.token_to_id[UNK_TOKEN]
        return [self.token_to_id.get(i, unk_id) for i in ingredients]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token.get(i, UNK_TOKEN) for i in ids]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.token_to_id, f, indent=2, sort_keys=False)

    @classmethod
    def load(cls, path: Path) -> "IngredientVocabulary":
        with open(path) as f:
            token_to_id = json.load(f)
        return cls(token_to_id)
