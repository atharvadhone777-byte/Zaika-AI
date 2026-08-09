"""
Cleaning and normalization for the Food.com recipes dataset.

Design principle: every function here does ONE transformation and is
independently unit-testable (see tests/test_preprocessing.py). This is
deliberate - "preprocessing" as a single 200-line function is exactly the
kind of code an interviewer will ask you to justify piece by piece, and
you want to be able to point at one function per decision.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

import pandas as pd

from ml.config import PREPROCESSING


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_stringified_list(value: str) -> list:
    """
    Food.com stores list-typed columns (ingredients, steps, tags, nutrition)
    as their Python repr inside a CSV cell, e.g. "['tomato', 'onion']".
    ast.literal_eval is used instead of eval() because it only parses
    literal Python data structures and cannot execute arbitrary code -
    the correct choice whenever you're parsing untrusted/scraped strings.
    """
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


NUTRITION_FIELDS = (
    "calories", "total_fat_pdv", "sugar_pdv", "sodium_pdv",
    "protein_pdv", "saturated_fat_pdv", "carbohydrates_pdv",
)


def expand_nutrition_column(df: pd.DataFrame) -> pd.DataFrame:
    """Expands the single 'nutrition' list column into 7 named numeric columns.

    Named columns (vs. keeping a list) are used because downstream code
    (the difficulty heuristic, the API's nutrition response, the EDA) all
    need to address specific fields - a list column forces every consumer
    to remember index 0 = calories, which is fragile and unreadable.
    """
    parsed = df["nutrition"].apply(parse_stringified_list)
    nutrition_df = pd.DataFrame(
        parsed.tolist(), columns=list(NUTRITION_FIELDS), index=df.index
    )
    return pd.concat([df.drop(columns=["nutrition"]), nutrition_df], axis=1)


# ---------------------------------------------------------------------------
# Ingredient text normalization
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")

# Common surface variations that refer to the same ingredient. Kept as an
# explicit dict rather than a stemmer: stemming ("tomatoes" -> "tomato") is
# fine for plurals, but a stemmer would also mangle unrelated words in
# recipe text, and can't fix cases like "scallions"/"green onions" being
# the same thing. An explicit mapping is more precise for a domain-specific,
# closed-ish vocabulary like cooking ingredients.
INGREDIENT_ALIASES = {
    "tomatoes": "tomato",
    "onions": "onion",
    "green onions": "scallion",
    "spring onions": "scallion",
    "garlic cloves": "garlic",
    "cloves garlic": "garlic",
    "eggs": "egg",
    "potatoes": "potato",
    "carrots": "carrot",
    "chicken breasts": "chicken breast",
    "boneless chicken breast": "chicken breast",
}


def normalize_ingredient(raw: str) -> str:
    """
    Normalizes a single ingredient string:
      1. lowercase
      2. strip parenthetical asides, e.g. "onion (chopped)" -> "onion"
      3. strip non-alphabetic characters (quantities/units that leaked in)
      4. collapse whitespace
      5. map known aliases to a canonical form

    Each step is intentionally conservative - the goal is to merge
    obvious duplicates (case, plural, common alias) without merging
    genuinely different ingredients (e.g. "green pepper" must stay
    distinct from "black pepper").
    """
    text = raw.lower().strip()
    text = re.sub(r"\([^)]*\)", "", text)          # remove "(chopped)" etc.
    text = _NON_ALPHA_RE.sub(" ", text)             # drop digits/punctuation
    text = _WHITESPACE_RE.sub(" ", text).strip()
    text = INGREDIENT_ALIASES.get(text, text)
    return text


def normalize_ingredient_list(raw_list: list[str]) -> list[str]:
    normalized = [normalize_ingredient(i) for i in raw_list]
    # dict.fromkeys instead of set() to normalize AND de-duplicate while
    # preserving the original order - order matters for reproducible token
    # sequences fed into the model later.
    deduped = list(dict.fromkeys(n for n in normalized if n))
    return [i for i in deduped if i not in PREPROCESSING.stopword_ingredients]


# ---------------------------------------------------------------------------
# Row-level filtering
# ---------------------------------------------------------------------------

@dataclass
class FilterReport:
    """Records why rows were dropped, so cleaning is auditable rather than
    a silent black box. This gets logged and is worth showing in the EDA
    notebook - "X% of rows dropped, here's why" is exactly what a reviewer
    wants to see before trusting a training set."""
    starting_rows: int
    dropped_null_critical_fields: int = 0
    dropped_duplicates: int = 0
    dropped_bad_minutes: int = 0
    dropped_too_few_steps: int = 0
    ending_rows: int = 0

    def summary(self) -> str:
        return (
            f"Started with {self.starting_rows} rows.\n"
            f"  - dropped {self.dropped_null_critical_fields} rows with null critical fields\n"
            f"  - dropped {self.dropped_duplicates} exact/near duplicate rows\n"
            f"  - dropped {self.dropped_bad_minutes} rows with implausible cook time\n"
            f"  - dropped {self.dropped_too_few_steps} rows with too few steps\n"
            f"Ending with {self.ending_rows} rows "
            f"({self.ending_rows / self.starting_rows:.1%} retained)."
        )


def clean_recipes(df: pd.DataFrame) -> tuple[pd.DataFrame, FilterReport]:
    """Full cleaning pipeline for the raw recipes dataframe. Returns the
    cleaned dataframe plus a report of what was removed and why."""
    report = FilterReport(starting_rows=len(df))
    df = df.copy()

    # 1. Critical fields must be present - a recipe with no ingredients or
    #    name isn't usable no matter how the rest of the pipeline is built.
    before = len(df)
    df = df.dropna(subset=["name", "ingredients", "steps"])
    report.dropped_null_critical_fields = before - len(df)

    # 2. Parse list-typed columns now that we know they're present.
    df["ingredients"] = df["ingredients"].apply(parse_stringified_list)
    df["steps"] = df["steps"].apply(parse_stringified_list)
    df["tags"] = df["tags"].apply(parse_stringified_list)

    # 3. Normalize ingredients (see normalize_ingredient_list above).
    df["ingredients"] = df["ingredients"].apply(normalize_ingredient_list)

    # 4. Duplicates: Food.com aggregates from multiple sources, so the same
    #    recipe (same name + same ingredient set) can appear more than once.
    #    We dedup on (name, sorted ingredients) rather than the raw row,
    #    because two rows can differ only in id/contributor/submission date
    #    while describing the identical recipe.
    before = len(df)
    df["_dedup_key"] = df["name"].str.lower().str.strip() + "|" + df["ingredients"].apply(lambda x: ",".join(sorted(x)))
    df = df.drop_duplicates(subset="_dedup_key").drop(columns="_dedup_key")
    report.dropped_duplicates = before - len(df)

    # 5. Implausible cook times are dropped, not clipped (see config.py
    #    docstring for why: clipping fabricates values, dropping doesn't).
    before = len(df)
    df = df[(df["minutes"] >= PREPROCESSING.min_minutes) & (df["minutes"] <= PREPROCESSING.max_minutes)]
    report.dropped_bad_minutes = before - len(df)

    # 6. Too few steps usually means a broken scrape ("see original site").
    before = len(df)
    df = df[df["n_steps"] >= PREPROCESSING.min_steps]
    report.dropped_too_few_steps = before - len(df)

    # 7. Expand nutrition into named columns now that rows are finalized.
    df = expand_nutrition_column(df)

    df = df.reset_index(drop=True)
    report.ending_rows = len(df)
    return df, report


# ---------------------------------------------------------------------------
# Difficulty heuristic
# ---------------------------------------------------------------------------

def estimate_difficulty(n_steps: int, minutes: float, n_ingredients: int) -> str:
    """
    Rule-based difficulty estimate, deliberately NOT a learned model.

    Rationale (worth stating explicitly in an interview): difficulty is a
    label the dataset doesn't provide, and training a second model to
    predict it would need ground-truth labels we don't have - we'd just be
    learning to reproduce our own heuristic with extra steps. A transparent,
    editable rule is more honest about what it is, and it's directly
    explainable to a user ("marked hard because it has 14 steps and takes
    90+ minutes") which a learned classifier's output would not be.
    """
    score = 0
    score += 1 if n_steps > 6 else 0
    score += 1 if n_steps > 12 else 0
    score += 1 if minutes > 45 else 0
    score += 1 if minutes > 90 else 0
    score += 1 if n_ingredients > 10 else 0

    if score <= 1:
        return "easy"
    elif score <= 3:
        return "medium"
    return "hard"
