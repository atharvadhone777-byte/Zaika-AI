"""
Trains the ingredient encoder via in-batch contrastive learning.

Usage:
    python -m ml.training.train                  # single run with config.py defaults
    python -m ml.training.train --sweep           # small hyperparameter sweep, logs a comparison table
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from ml.config import (
    CLEANED_RECIPES_PATH, TRAIN_SPLIT_PATH, VAL_SPLIT_PATH, VOCAB_PATH,
    MODELS_DIR, DOCS_DIR, RANDOM_SEED, PREPROCESSING, MODEL, TRAINING,
)
from ml.data.dataset import make_training_pairs, make_tf_dataset
from ml.data.tokenizer import IngredientVocabulary
from ml.models.encoder import build_encoder, build_siamese_training_model
from ml.training.losses import make_in_batch_infonce_loss

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def _load_split(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["ingredients"] = df["ingredients"].apply(json.loads)
    return df


def build_callbacks(run_dir: Path, patience_es: int, patience_lr: int, lr_factor: float) -> list:
    """
    All four callbacks the assignment requires, each doing a distinct job:
      - EarlyStopping: stop once val_loss stops improving, restore the best
        weights seen (not the last epoch's, which may already be overfit).
      - ModelCheckpoint: persist the best epoch to disk independently of
        whether the run is later interrupted.
      - ReduceLROnPlateau: shrink the learning rate when progress stalls,
        which typically lets the model settle into a sharper minimum than
        a fixed LR would, without needing a hand-tuned decay schedule.
      - TensorBoard: per-epoch scalars for later inspection.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience_es, restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "checkpoint.weights.h5"),
            monitor="val_loss", save_best_only=True, save_weights_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=lr_factor, patience=patience_lr, min_lr=1e-6,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(run_dir / "tensorboard")),
    ]


def train_one_config(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    vocab: IngredientVocabulary,
    embedding_dim: int,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    run_name: str,
) -> dict:
    max_len = MODEL.max_ingredients_per_recipe

    train_q, train_p = make_training_pairs(train_df, vocab, max_len)
    val_q, val_p = make_training_pairs(val_df, vocab, max_len)

    # With a very small dataset, an unlucky batch_size can exceed the
    # number of available pairs; clamp so the sweep never crashes on the
    # sample data. This has no effect once run against the full dataset.
    effective_batch = min(batch_size, len(train_q))
    train_ds = make_tf_dataset(train_q, train_p, batch_size=effective_batch, shuffle=True)
    val_ds = make_tf_dataset(val_q, val_p, batch_size=min(batch_size, max(len(val_q), 1)), shuffle=False)

    encoder = build_encoder(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        max_len=max_len,
        hidden_units=MODEL.dense_hidden_units,
        dropout_rate=MODEL.dropout_rate,
    )
    trainer = build_siamese_training_model(encoder, max_len=max_len)
    trainer.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=make_in_batch_infonce_loss(temperature=0.07),
    )

    run_dir = MODELS_DIR / "runs" / run_name
    callbacks = build_callbacks(
        run_dir, TRAINING.early_stopping_patience, TRAINING.reduce_lr_patience, TRAINING.reduce_lr_factor
    )

    start = time.time()
    history = trainer.fit(
        train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=2,
    )
    elapsed = time.time() - start

    best_val_loss = min(history.history["val_loss"])
    n_params = trainer.count_params()

    return {
        "run_name": run_name,
        "embedding_dim": embedding_dim,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs_run": len(history.history["loss"]),
        "best_val_loss": round(float(best_val_loss), 4),
        "final_train_loss": round(float(history.history["loss"][-1]), 4),
        "n_params": int(n_params),
        "train_seconds": round(elapsed, 1),
        "encoder": encoder,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="run a small hyperparameter sweep instead of a single training run")
    parser.add_argument("--epochs", type=int, default=TRAINING.epochs)
    args = parser.parse_args()

    vocab = IngredientVocabulary.load(VOCAB_PATH)
    train_df = _load_split(TRAIN_SPLIT_PATH)
    val_df = _load_split(VAL_SPLIT_PATH)
    print(f"train: {len(train_df)} recipes, val: {len(val_df)} recipes, vocab: {len(vocab)} tokens")

    if args.sweep:
        # Small, deliberately cheap grid: embedding_dim controls model
        # capacity, learning_rate controls optimization stability. Both are
        # the two hyperparameters most likely to matter for a model this
        # size - a wider sweep (dropout, hidden layer sizes, temperature)
        # is listed as future work in the README rather than run here,
        # because each additional axis multiplies runtime and this dataset
        # is too small for a fine-grained sweep to be meaningful anyway.
        grid = [
            {"embedding_dim": 64, "learning_rate": 1e-3},
            {"embedding_dim": 128, "learning_rate": 1e-3},
            {"embedding_dim": 128, "learning_rate": 5e-4},
        ]
        results = []
        for i, cfg in enumerate(grid):
            print(f"\n=== sweep run {i+1}/{len(grid)}: {cfg} ===")
            result = train_one_config(
                train_df, val_df, vocab,
                embedding_dim=cfg["embedding_dim"],
                learning_rate=cfg["learning_rate"],
                batch_size=TRAINING.batch_size,
                epochs=args.epochs,
                run_name=f"sweep_{i}_dim{cfg['embedding_dim']}_lr{cfg['learning_rate']}",
            )
            results.append(result)

        best = min(results, key=lambda r: r["best_val_loss"])
        print("\n=== Sweep results (sorted by best_val_loss) ===")
        table_rows = sorted(results, key=lambda r: r["best_val_loss"])
        for r in table_rows:
            print(f"  {r['run_name']:35s}  val_loss={r['best_val_loss']:.4f}  "
                  f"params={r['n_params']:,}  time={r['train_seconds']}s")
        print(f"\nBest config: {best['run_name']} (val_loss={best['best_val_loss']})")

        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        with open(DOCS_DIR / "hyperparameter_sweep.json", "w") as f:
            json.dump(
                [{k: v for k, v in r.items() if k != "encoder"} for r in results],
                f, indent=2,
            )

        _export_encoder(best["encoder"])

    else:
        result = train_one_config(
            train_df, val_df, vocab,
            embedding_dim=MODEL.embedding_dim,
            learning_rate=TRAINING.learning_rate,
            batch_size=TRAINING.batch_size,
            epochs=args.epochs,
            run_name="single_run",
        )
        print(f"\nFinal val_loss: {result['best_val_loss']}, params: {result['n_params']:,}")
        _export_encoder(result["encoder"])


def _export_encoder(encoder: tf.keras.Model) -> None:
    """
    Exports in both formats the assignment requires. Keras 3 (bundled with
    TF 2.16+) changed how these work, worth being able to explain:
      - `.export()` writes a TF SavedModel containing an inference-only
        serving signature - the right artifact for TF Serving or a plain
        `tf.saved_model.load()` call, but it is NOT reloadable as a Keras
        Model (no optimizer state, no re-compilation).
      - `.save(..., .h5)` writes the full Keras model (architecture +
        weights) and IS reloadable via `tf.keras.models.load_model()` -
        this is what ml/inference/predictor.py actually loads, since
        inference code wants a live Keras Model to call `.predict()` on.
    Both are produced so the deliverable satisfies "SavedModel + H5", but
    in practice the H5 file is the one the rest of this project consumes.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    encoder.export(str(MODELS_DIR / "encoder_savedmodel"))
    encoder.save(str(MODELS_DIR / "encoder.h5"))
    print(f"Encoder exported to {MODELS_DIR} (SavedModel + H5)")


if __name__ == "__main__":
    main()
