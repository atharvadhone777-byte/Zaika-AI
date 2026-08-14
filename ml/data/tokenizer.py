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
