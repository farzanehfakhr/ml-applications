"""
CIFAR-10 CNN: architecture/hyperparameter ablation study.

Starting point: a standard 2-conv-block CNN on a 30% subset of CIFAR-10
(the full dataset was reduced deliberately, to leave more headroom for
visible improvement - see README for the reasoning). Each change below
was added one at a time and measured against the previous step.

This script assembles the *final, best* configuration found during the
ablation (BatchNorm + Dropout(0.3) + wider dense layer + an extra Conv2D
block + a tuned learning rate + more epochs). See the README's results
table for the accuracy at every intermediate step.

Reconstructed from the author's own notebook screenshots (MET CS 767,
Assignment 3, Section 1). Transcribed, not re-derived.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, models, layers


# ----------- Data -----------
# CIFAR-10 is one of a few built-in Keras datasets
# (see https://www.cs.toronto.edu/~kriz/cifar.html)
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Use only 30% of training data to make improvement room
subset = int(len(train_images) * 0.3)
train_images, train_labels = train_images[:subset], train_labels[:subset]

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


# ----------- Final architecture (all improvements combined) -----------
def build_improved_model():
    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 3
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())

    # Extra Conv2D layer (deeper feature extraction before flattening)
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0008),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    return model


if __name__ == "__main__":
    model = build_improved_model()
    history = model.fit(
        train_images, train_labels,
        epochs=5,
        validation_data=(test_images, test_labels),
    )

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print("Final test accuracy:", test_acc)
    # Author's result with this configuration: ~0.61 (vs. 0.48 reduced-data
    # baseline and 0.60 on the original full-data, un-reduced baseline)
