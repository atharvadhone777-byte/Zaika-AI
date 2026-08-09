"""
Unit tests for ml/data/preprocessing.py.

These target the preprocessing logic specifically (not the API, not the
model) - deliberately fast, dependency-free tests that run in under a
second and don't need the dataset on disk. Each test targets exactly one
documented decision from preprocessing.py's docstrings, so a reviewer can
match test -> decision -> justification directly.
"""

import pandas as pd
import pytest

from ml.data.preprocessing import (
    clean_recipes,
    estimate_difficulty,
    normalize_ingredient,
    normalize_ingredient_list,
    parse_stringified_list,
)


def test_parse_stringified_list_valid():
    assert parse_stringified_list("['tomato', 'onion']") == ["tomato", "onion"]


def test_parse_stringified_list_handles_malformed_input_safely():
    # Must not raise and must not execute arbitrary code - this is the
    # entire point of using ast.literal_eval over eval().
    assert parse_stringified_list("not a list") == []
    assert parse_stringified_list("__import__('os').system('echo pwned')") == []


def test_parse_stringified_list_nan():
    assert parse_stringified_list(float("nan")) == []


@pytest.mark.parametrize("raw,expected", [
    ("Tomatoes", "tomato"),
    ("  Onion (chopped)  ", "onion"),
    ("green onions", "scallion"),
    ("GARLIC CLOVES", "garlic"),
    ("black pepper", "black pepper"),  # must NOT collapse into "pepper"
])
def test_normalize_ingredient(raw, expected):
    assert normalize_ingredient(raw) == expected


def test_normalize_ingredient_list_dedupes_and_strips_stopwords():
    raw = ["Tomatoes", "tomato", "Salt", "onion"]
    result = normalize_ingredient_list(raw)
    assert result == ["tomato", "onion"]  # order preserved, salt dropped, dupes merged


def test_clean_recipes_drops_bad_minutes_and_reports_it():
    df = pd.DataFrame([
        {"id": 1, "name": "Good recipe", "minutes": 30, "n_steps": 3,
         "ingredients": "['tomato', 'onion']", "steps": "['step1', 'step2', 'step3']",
         "tags": "[]", "nutrition": "[100,1,1,1,1,1,1]", "n_ingredients": 2},
        {"id": 2, "name": "Bad minutes", "minutes": -5, "n_steps": 3,
         "ingredients": "['tomato']", "steps": "['step1','step2','step3']",
         "tags": "[]", "nutrition": "[100,1,1,1,1,1,1]", "n_ingredients": 1},
        {"id": 3, "name": "Too few steps", "minutes": 20, "n_steps": 1,
         "ingredients": "['tomato']", "steps": "['step1']",
         "tags": "[]", "nutrition": "[100,1,1,1,1,1,1]", "n_ingredients": 1},
    ])
    clean, report = clean_recipes(df)
    assert len(clean) == 1
    assert clean.iloc[0]["name"] == "Good recipe"
    assert report.dropped_bad_minutes == 1
    assert report.dropped_too_few_steps == 1


def test_clean_recipes_dedupes_same_name_and_ingredient_set():
    df = pd.DataFrame([
        {"id": 1, "name": "Tomato Rice", "minutes": 30, "n_steps": 3,
         "ingredients": "['tomato', 'rice']", "steps": "['a','b','c']",
         "tags": "[]", "nutrition": "[100,1,1,1,1,1,1]", "n_ingredients": 2},
        {"id": 2, "name": "tomato rice", "minutes": 25, "n_steps": 4,  # same dish, different id/casing/minutes
         "ingredients": "['rice', 'tomato']", "steps": "['a','b','c','d']",
         "tags": "[]", "nutrition": "[100,1,1,1,1,1,1]", "n_ingredients": 2},
    ])
    clean, report = clean_recipes(df)
    assert len(clean) == 1
    assert report.dropped_duplicates == 1


def test_estimate_difficulty_thresholds():
    assert estimate_difficulty(n_steps=3, minutes=15, n_ingredients=4) == "easy"
    assert estimate_difficulty(n_steps=8, minutes=50, n_ingredients=6) == "medium"
    assert estimate_difficulty(n_steps=15, minutes=120, n_ingredients=14) == "hard"
