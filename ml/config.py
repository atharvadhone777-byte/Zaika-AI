"""
Central configuration for the ML pipeline.

Why a single config module instead of scattering constants across files:
every script in ml/ (preprocessing, training, evaluation, inference) needs
the same paths and the same random seed. Defining them once means changing
a path or the seed happens in exactly one place, and it makes the pipeline
reproducible end-to-end - a reviewer can read this file and know exactly
what "the config" was for any given run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DATA_SPLITS_DIR = ROOT_DIR / "data" / "splits"
MODELS_DIR = ROOT_DIR / "models" / "v1"
DOCS_DIR = ROOT_DIR / "docs"

RAW_RECIPES_PATH = DATA_RAW_DIR / "RAW_recipes.csv"
RAW_INTERACTIONS_PATH = DATA_RAW_DIR / "RAW_interactions.csv"
CLEANED_RECIPES_PATH = DATA_PROCESSED_DIR / "recipes_clean.parquet"

TRAIN_SPLIT_PATH = DATA_SPLITS_DIR / "train.parquet"
VAL_SPLIT_PATH = DATA_SPLITS_DIR / "val.parquet"
TEST_SPLIT_PATH = DATA_SPLITS_DIR / "test.parquet"

VOCAB_PATH = MODELS_DIR / "ingredient_vocab.json"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

# One seed reused everywhere (numpy, tensorflow, python's random, and the
# train/val/test split) so results are reproducible run to run. This is
# worth stating explicitly in an interview: reproducibility isn't automatic
# in ML pipelines, it's a deliberate design choice enforced at every stage
# that touches randomness.
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreprocessingConfig:
    # Ingredients that appear in Food.com's data but carry ~no discriminative
    # signal for "what can I cook" (salt, water, pepper appear in almost
    # every recipe). Stripping them prevents the embedding model from
    # wasting capacity on near-constant features. Kept as an explicit,
    # editable list rather than a frequency cutoff so the decision is
    # auditable rather than a black-box threshold.
    stopword_ingredients: tuple[str, ...] = (
        "salt", "water", "pepper", "black pepper", "ground black pepper",
    )

    # Ingredients with a document frequency below this count are dropped
    # from the vocabulary (treated as <UNK>). Below this threshold there
    # isn't enough co-occurrence signal for the embedding model to learn
    # anything reliable about the ingredient.
    min_ingredient_frequency: int = 5

    # Recipes with fewer than this many steps are usually malformed scrapes
    # (e.g. "see website") rather than real recipes, so they're dropped.
    min_steps: int = 2

    # Recipes with an implausible cook time are dropped rather than clipped,
    # because clipping would silently fabricate a plausible-looking but
    # false value; dropping is the honest choice for data we can't trust.
    min_minutes: int = 1
    max_minutes: int = 24 * 60  # 1 day - anything longer is almost always a data error


@dataclass(frozen=True)
class SplitConfig:
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1


@dataclass(frozen=True)
class ModelConfig:
    embedding_dim: int = 128
    vocab_size: int = 8000          # set dynamically from the real vocab at train time
    max_ingredients_per_recipe: int = 20
    dense_hidden_units: tuple[int, ...] = (256, 128)
    dropout_rate: float = 0.2


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 256
    epochs: int = 50
    learning_rate: float = 1e-3
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    triplet_margin: float = 0.3     # margin for the contrastive/triplet loss, see ml/training/train.py


PREPROCESSING = PreprocessingConfig()
SPLIT = SplitConfig()
MODEL = ModelConfig()
TRAINING = TrainingConfig()
