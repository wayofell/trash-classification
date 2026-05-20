"""
Model architectures for Garbage Classification.

Provides:
- build_baseline_cnn(): simple CNN from scratch
- (later) build_resnet50_transfer(): ResNet50 with ImageNet weights
- (later) build_efficientnet_transfer(): EfficientNetV2S with ImageNet weights

All models accept input of shape (224, 224, 3) and output NUM_CLASSES logits.
"""

from typing import Optional

import tensorflow as tf
from tensorflow import keras
from keras import layers


# Match constants from src/data.py
IMG_SIZE = (224, 224)
NUM_CLASSES = 10


# ---------------------------------------------------------------------------
# Baseline CNN
# ---------------------------------------------------------------------------

def build_baseline_cnn(
        num_classes: int = NUM_CLASSES,
        input_shape: tuple = (*IMG_SIZE, 3),
        dropout_rate: float = 0.5,
        name: str = "baseline_cnn",
) -> keras.Model:
    """
    Simple CNN trained from scratch.

    Architecture:
        4 convolutional blocks: Conv -> BN -> ReLU -> MaxPool
        Filters: 32 -> 64 -> 128 -> 256
        Global average pooling
        Dense classifier with dropout

    This serves as the baseline reference point. Transfer learning models
    should significantly outperform it.
    """
    inputs = keras.Input(shape=input_shape, name="input")

    # Normalize from [0, 255] to [0, 1]
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)

    # Block 1: 224 -> 112
    x = layers.Conv2D(32, 3, padding="same", use_bias=False, name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D(2, name="pool1")(x)

    # Block 2: 112 -> 56
    x = layers.Conv2D(64, 3, padding="same", use_bias=False, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D(2, name="pool2")(x)

    # Block 3: 56 -> 28
    x = layers.Conv2D(128, 3, padding="same", use_bias=False, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling2D(2, name="pool3")(x)

    # Block 4: 28 -> 14
    x = layers.Conv2D(256, 3, padding="same", use_bias=False, name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.ReLU(name="relu4")(x)
    x = layers.MaxPooling2D(2, name="pool4")(x)

    # Global pooling instead of Flatten — fewer parameters, less overfitting
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Classifier head
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, name="logits")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


# ---------------------------------------------------------------------------
# ResNet50 Transfer Learning
# ---------------------------------------------------------------------------

def build_resnet50_transfer(
        num_classes: int = NUM_CLASSES,
        input_shape: tuple = (*IMG_SIZE, 3),
        dropout_rate: float = 0.3,
        trainable_base: bool = False,
        name: str = "resnet50_transfer",
) -> keras.Model:
    """
    ResNet50 pre-trained on ImageNet with a custom classifier head.

    Args:
        trainable_base: if False (default), ResNet50 weights are frozen.
                        Set to True for fine-tuning.

    The model includes ImageNet-style preprocessing:
        input pixels in [0, 255] -> ResNet preprocessing (BGR + ImageNet mean)
    """
    inputs = keras.Input(shape=input_shape, name="input")

    # ResNet expects BGR images with ImageNet mean subtracted, in [0, 255] range.
    # We use the official preprocessing layer to match training conditions.
    x = keras.applications.resnet50.preprocess_input(inputs)

    # Load ResNet50 with ImageNet weights, no top (no classifier)
    base = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling=None,
        name="resnet50_base",
    )
    base.trainable = trainable_base

    # Important: keep BatchNorm layers in inference mode even if base is unfrozen.
    # Otherwise BN stats would update on the small new dataset and break the
    # pre-trained features.
    x = base(x, training=False)

    # Classifier head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, name="logits")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def unfreeze_last_block(model: keras.Model, block_name: str = "conv5_block3") -> int:
    """
    Unfreeze the last block of ResNet50 base for fine-tuning.

    Walks through layers of the inner 'resnet50_base' submodel and sets
    layer.trainable = True for layers whose name contains block_name.
    BatchNorm layers within the block are kept frozen to preserve
    ImageNet running statistics.

    Returns the number of layers unfrozen.
    """
    base = model.get_layer("resnet50_base")
    base.trainable = True  # required at outer level

    unfrozen = 0
    for layer in base.layers:
        if block_name in layer.name and not isinstance(
                layer, layers.BatchNormalization
        ):
            layer.trainable = True
            unfrozen += 1
        else:
            layer.trainable = False

    return unfrozen


# ---------------------------------------------------------------------------
# EfficientNetV2S Transfer Learning
# ---------------------------------------------------------------------------

def build_efficientnetv2s_transfer(
        num_classes: int = NUM_CLASSES,
        input_shape: tuple = (*IMG_SIZE, 3),
        dropout_rate: float = 0.3,
        trainable_base: bool = False,
        name: str = "efficientnetv2s_transfer",
) -> keras.Model:
    """
    EfficientNetV2S pre-trained on ImageNet with a custom classifier head.

    Unlike ResNet50, EfficientNetV2 expects raw [0, 255] uint8 / float input;
    its preprocessing (normalization to [-1, 1]) is built into the model.
    No external preprocess_input call is needed.

    Args:
        trainable_base: if False (default), EfficientNetV2S weights are frozen.
    """
    inputs = keras.Input(shape=input_shape, name="input")

    base = keras.applications.EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling=None,
        include_preprocessing=True,  # builds preprocessing into the graph
    )
    base._name = "efficientnetv2s_base"  # rename after construction
    base.trainable = trainable_base

    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, name="logits")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Baseline CNN ===")
    baseline = build_baseline_cnn()
    print(f"Total parameters: {baseline.count_params():,}\n")

    print("=== ResNet50 Transfer (frozen) ===")
    resnet = build_resnet50_transfer(trainable_base=False)
    trainable = sum(
        keras.backend.count_params(w) for w in resnet.trainable_weights
    )
    print(f"Total parameters: {resnet.count_params():,}")
    print(f"Trainable parameters (frozen): {trainable:,}\n")

    print("=== EfficientNetV2S Transfer (frozen) ===")
    effnet = build_efficientnetv2s_transfer(trainable_base=False)
    trainable = sum(
        keras.backend.count_params(w) for w in effnet.trainable_weights
    )
    print(f"Total parameters: {effnet.count_params():,}")
    print(f"Trainable parameters (frozen): {trainable:,}")