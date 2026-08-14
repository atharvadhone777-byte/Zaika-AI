from __future__ import annotations

import ast
import re
from dataclasses import dataclass

import pandas as pd

from ml.config import PREPROCESSING

def parse_stringified_list(value: str) -> list:
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
    parsed = df["nutrition"].apply(parse_stringified_list)
    nutrition_df = pd.DataFrame(
        parsed.tolist(), columns=list(NUTRITION_FIELDS), index=df.index
    )
    return pd.concat([df.drop(columns=["nutrition"]), nutrition_df], axis=1)

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")

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
    text = raw.lower().strip()
    text = re.sub(r"\([^)]*\)", "", text)          
    text = _NON_ALPHA_RE.sub(" ", text)             
    text = _WHITESPACE_RE.sub(" ", text).strip()
    text = INGREDIENT_ALIASES.get(text, text)
    return text


def normalize_ingredient_list(raw_list: list[str]) -> list[str]:
    normalized = [normalize_ingredient(i) for i in raw_list]
    deduped = list(dict.fromkeys(n for n in normalized if n))
    return [i for i in deduped if i not in PREPROCESSING.stopword_ingredients]

@dataclass
class FilterReport:
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
    report = FilterReport(starting_rows=len(df))
    df = df.copy()

    before = len(df)
    df = df.dropna(subset=["name", "ingredients", "steps"])
    report.dropped_null_critical_fields = before - len(df)

    df["ingredients"] = df["ingredients"].apply(parse_stringified_list)
    df["steps"] = df["steps"].apply(parse_stringified_list)
    df["tags"] = df["tags"].apply(parse_stringified_list)

    df["ingredients"] = df["ingredients"].apply(normalize_ingredient_list)

    before = len(df)
    df["_dedup_key"] = df["name"].str.lower().str.strip() + "|" + df["ingredients"].apply(lambda x: ",".join(sorted(x)))
    df = df.drop_duplicates(subset="_dedup_key").drop(columns="_dedup_key")
    report.dropped_duplicates = before - len(df)

    before = len(df)
    df = df[(df["minutes"] >= PREPROCESSING.min_minutes) & (df["minutes"] <= PREPROCESSING.max_minutes)]
    report.dropped_bad_minutes = before - len(df)

    before = len(df)
    df = df[df["n_steps"] >= PREPROCESSING.min_steps]
    report.dropped_too_few_steps = before - len(df)

    df = expand_nutrition_column(df)

    df = df.reset_index(drop=True)
    report.ending_rows = len(df)
    return df, report

def estimate_difficulty(n_steps: int, minutes: float, n_ingredients: int) -> str:
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
