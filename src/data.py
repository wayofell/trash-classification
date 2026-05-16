"""
Data pipeline for Garbage Classification V2.

Provides:
- load_dataset(): builds train/val/test tf.data.Dataset objects
- get_class_weights(): computes inverse-frequency class weights
- get_class_names(): returns alphabetically sorted class names

All functions are environment-aware: paths resolve correctly both on
Kaggle (/kaggle/input/...) and locally (./data/sample/ for quick tests).
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KAGGLE_DATA_ROOT = Path(
    '/kaggle/input/datasets/sumn2u/garbage-classification-v2/standardized_256'
)
LOCAL_DATA_ROOT = Path('./data/sample')

INPUT_SIZE = (256, 256)
IMG_SIZE = (224, 224)
NUM_CLASSES = 10
CLASS_NAMES = [
    'battery', 'biological', 'cardboard', 'clothes', 'glass',
    'metal', 'paper', 'plastic', 'shoes', 'trash'
]

DEFAULT_SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def get_data_root() -> Path:
    """Return the dataset root for the current environment."""
    for candidate in [KAGGLE_DATA_ROOT, LOCAL_DATA_ROOT]:
        if candidate.exists() and (candidate / CLASS_NAMES[0]).exists():
            return candidate
    raise FileNotFoundError(
        f"Dataset not found or empty. Checked:\n"
        f"  Kaggle: {KAGGLE_DATA_ROOT}\n"
        f"  Local:  {LOCAL_DATA_ROOT}\n"
        f"Each path must contain class subdirectories: {CLASS_NAMES}"
    )


def get_class_names() -> List[str]:
    """Return alphabetically sorted class names."""
    return list(CLASS_NAMES)


# ---------------------------------------------------------------------------
# File listing with stratified split
# ---------------------------------------------------------------------------

def _list_files_per_class(data_root: Path) -> Dict[str, List[Path]]:
    """Return {class_name: [file paths sorted]} for reproducibility."""
    result = {}
    for cls in CLASS_NAMES:
        cls_dir = data_root / cls
        files = sorted(cls_dir.iterdir())
        result[cls] = files
    return result


def _stratified_split(
        files_per_class: Dict[str, List[Path]],
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = DEFAULT_SEED,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """
    Stratified split: each class is split independently with the same ratios.
    Returns three lists of (file_path, class_index) tuples.
    """
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []

    for class_idx, cls in enumerate(CLASS_NAMES):
        files = files_per_class[cls]
        n = len(files)
        indices = rng.permutation(n)

        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)

        test_idx = indices[:n_test]
        val_idx = indices[n_test:n_test + n_val]
        train_idx = indices[n_test + n_val:]

        for i in train_idx:
            train.append((files[i], class_idx))
        for i in val_idx:
            val.append((files[i], class_idx))
        for i in test_idx:
            test.append((files[i], class_idx))

    # Shuffle each split (paths within split, not classes)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


# ---------------------------------------------------------------------------
# Image loading and preprocessing
# ---------------------------------------------------------------------------

def _decode_image(path: tf.Tensor) -> tf.Tensor:
    """Load a JPEG image from disk and resize to INPUT_SIZE (256x256)."""
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, INPUT_SIZE, method='bilinear')
    return img


def _train_preprocess(img: tf.Tensor) -> tf.Tensor:
    """Random crop + augmentations for training."""
    img = tf.image.random_crop(img, size=(*IMG_SIZE, 3))
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.15 * 255)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img


def _eval_preprocess(img: tf.Tensor) -> tf.Tensor:
    """Deterministic center crop for validation and test."""
    offset_h = (INPUT_SIZE[0] - IMG_SIZE[0]) // 2
    offset_w = (INPUT_SIZE[1] - IMG_SIZE[1]) // 2
    img = tf.image.crop_to_bounding_box(img, offset_h, offset_w, *IMG_SIZE)
    return img


def _build_dataset(
        samples: List[Tuple[Path, int]],
        training: bool,
        batch_size: int,
        seed: int = DEFAULT_SEED,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset from a list of (path, label) tuples."""
    paths = [str(p) for p, _ in samples]
    labels = [lbl for _, lbl in samples]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(buffer_size=len(samples), seed=seed,
                        reshuffle_each_iteration=True)

    def load_and_preprocess(path, label):
        img = _decode_image(path)
        img = _train_preprocess(img) if training else _eval_preprocess(img)
        return img, label

    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(
        batch_size: int = 32,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = DEFAULT_SEED,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[str, int]]:
    """
    Build train/val/test tf.data.Dataset objects with stratified split.

    Returns:
        train_ds: training dataset with augmentation
        val_ds:   validation dataset (deterministic preprocessing)
        test_ds:  test dataset (deterministic preprocessing)
        sizes:    dict with sample counts per split
    """
    data_root = get_data_root()
    files_per_class = _list_files_per_class(data_root)

    train, val, test = _stratified_split(
        files_per_class, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )

    train_ds = _build_dataset(train, training=True,
                              batch_size=batch_size, seed=seed)
    val_ds = _build_dataset(val, training=False, batch_size=batch_size)
    test_ds = _build_dataset(test, training=False, batch_size=batch_size)

    sizes = {'train': len(train), 'val': len(val), 'test': len(test)}
    return train_ds, val_ds, test_ds, sizes


def get_class_weights(
        files_per_class: Dict[str, List[Path]] = None,
) -> Dict[int, float]:
    """
    Compute class weights as inverse frequency, normalized so the mean is 1.0.

    Useful for handling class imbalance:
        model.fit(..., class_weight=get_class_weights())
    """
    if files_per_class is None:
        files_per_class = _list_files_per_class(get_data_root())

    counts = np.array([len(files_per_class[c]) for c in CLASS_NAMES])
    total = counts.sum()
    n_classes = len(CLASS_NAMES)

    # Standard sklearn formula: total / (n_classes * count)
    weights = total / (n_classes * counts)
    return {i: float(w) for i, w in enumerate(weights)}


# ---------------------------------------------------------------------------
# Smoke test (run when executed as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Smoke test: loading data...")
    train_ds, val_ds, test_ds, sizes = load_dataset(batch_size=8)

    print(f"\nSplit sizes:")
    for split, n in sizes.items():
        print(f"  {split:5s}: {n:>5d} samples")

    print(f"\nFirst training batch:")
    for images, labels in train_ds.take(1):
        print(f"  images shape: {images.shape}")
        print(f"  images dtype: {images.dtype}")
        print(f"  images range: [{tf.reduce_min(images):.1f}, "
              f"{tf.reduce_max(images):.1f}]")
        print(f"  labels shape: {labels.shape}")
        print(f"  labels sample: {labels.numpy()[:8].tolist()}")

    print(f"\nClass weights:")
    weights = get_class_weights()
    for i, cls in enumerate(CLASS_NAMES):
        print(f"  {cls:11s} ({i}): {weights[i]:.3f}")

    print("\nSmoke test passed.")