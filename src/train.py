"""
Training loop with standard callbacks.

Used by all experiment notebooks to keep training behavior consistent.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tensorflow as tf
from tensorflow import keras


def get_default_callbacks(
        checkpoint_path: Optional[Path] = None,
        early_stopping_patience: int = 5,
        lr_reduce_patience: int = 3,
) -> List[keras.callbacks.Callback]:
    """
    Standard callback set used across all experiments:

    - EarlyStopping: stop if val_accuracy doesn't improve for `patience` epochs,
                    restore best weights at the end
    - ReduceLROnPlateau: drop LR by 0.5 if val_loss plateaus
    - ModelCheckpoint: save best model by val_accuracy (optional)
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=early_stopping_patience,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=lr_reduce_patience,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    if checkpoint_path is not None:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
                save_weights_only=False,
                verbose=0,
            )
        )

    return callbacks


def compile_and_train(
        model: keras.Model,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        epochs: int = 30,
        learning_rate: float = 1e-3,
        optimizer_name: str = "adam",
        class_weight: Optional[Dict[int, float]] = None,
        callbacks: Optional[List[keras.callbacks.Callback]] = None,
        verbose: int = 1,
) -> Tuple[keras.callbacks.History, float]:
    """
    Compile the model and run training.

    Returns:
        history: Keras History object
        training_time_sec: wall-clock time in seconds
    """
    # Select optimizer
    if optimizer_name.lower() == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == "adamw":
        optimizer = keras.optimizers.AdamW(learning_rate=learning_rate)
    elif optimizer_name.lower() == "sgd":
        optimizer = keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=0.9, nesterov=True
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Compile
    # Note: from_logits=True because models output raw logits (no softmax)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    if callbacks is None:
        callbacks = get_default_callbacks()

    # Train
    start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=verbose,
    )
    elapsed = time.time() - start
    return history, elapsed


def evaluate_on_test(
        model: keras.Model,
        test_ds: tf.data.Dataset,
) -> Tuple[float, float]:
    """Return (test_loss, test_accuracy)."""
    results = model.evaluate(test_ds, verbose=0, return_dict=True)
    return float(results["loss"]), float(results["accuracy"])