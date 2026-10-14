"""
RecipeGAN: a conditional GAN that predicts a plausible ingredient list
directly from a photo of a dish.

Given a food image, an image encoder (CNN) produces a feature vector. A
generator combines that feature vector with random noise to produce a
multi-hot vector over a fixed ingredient vocabulary (TOP_K_INGREDIENTS).
A discriminator learns to distinguish real (image, ingredient-vector)
pairs from generated ones, pushing the generator toward image-consistent,
realistic-looking ingredient sets.

A supervised CNN baseline (same encoder, trained directly on binary
cross-entropy against the true ingredient vectors, no adversarial
training) is included for comparison - see the README for why it ended
up more reliable in practice than the GAN.

Dataset: "Food Ingredients and Recipes Dataset with Images" (Kaggle,
~14,000 images): https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images

Reconstructed from the author's own notebook screenshots (MET CS 767,
Assignment 4, Section 2: "Your GAN Application"). Transcribed, not
re-derived. The dataset, `df_model` (with an "image_path" column) and
`ingredient_vectors` / `ingredient_vocab` are not included here - swap in
your own.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image


IMG_SIZE = 128
IMG_FEATURE_DIM = 256
NOISE_DIM = 64
TOP_K_INGREDIENTS = 200
BATCH_SIZE = 32


# ----------- 1. Data pipeline -----------
def load_and_preprocess_image(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32)
    img = (img / 127.5) - 1.0  # normalize to [-1, 1]
    return img


def build_dataset(df_model, ingredient_vectors, batch_size=BATCH_SIZE):
    """
    df_model: dataframe with an "image_path" column
    ingredient_vectors: (N, TOP_K_INGREDIENTS) multi-hot array, same row
        order as df_model
    """
    paths_tensor = tf.constant(df_model["image_path"].values)
    labels_tensor = tf.constant(ingredient_vectors)

    dataset = tf.data.Dataset.from_tensor_slices((paths_tensor, labels_tensor))

    def map_fn(path, label):
        img = load_and_preprocess_image(path)
        return img, label

    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# ----------- 2. Image encoder -----------
def build_image_encoder():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Conv2D(32, 4, strides=2, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 4, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 4, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, 4, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(IMG_FEATURE_DIM, activation="relu")(x)
    return models.Model(inputs, x, name="image_encoder")


# ----------- 3. Generator -----------
def build_generator():
    noise_input = layers.Input(shape=(NOISE_DIM,))
    img_feat_input = layers.Input(shape=(IMG_FEATURE_DIM,))

    x = layers.Concatenate()([noise_input, img_feat_input])
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    out = layers.Dense(TOP_K_INGREDIENTS, activation="sigmoid")(x)

    return models.Model([noise_input, img_feat_input], out, name="generator")


# ----------- 4. Discriminator -----------
def build_discriminator():
    img_feat_input = layers.Input(shape=(IMG_FEATURE_DIM,))
    ing_input = layers.Input(shape=(TOP_K_INGREDIENTS,))

    x = layers.Concatenate()([img_feat_input, ing_input])
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    return models.Model([img_feat_input, ing_input], out, name="discriminator")


# ----------- 5. RecipeGAN training loop -----------
bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)


class RecipeGAN(tf.keras.Model):
    def __init__(self, encoder, generator, discriminator, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, gen_optimizer, disc_optimizer, **kwargs):
        super().compile(**kwargs)
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.bce = bce

    def train_step(self, data):
        images, real_ings = data
        batch_size = tf.shape(images)[0]

        # 1) Encode images
        img_feats = self.encoder(images, training=True)

        # 2) Train Discriminator
        noise = tf.random.normal((batch_size, NOISE_DIM))
        fake_ings = self.generator([noise, img_feats], training=True)

        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))

        with tf.GradientTape() as tape_disc:
            pred_real = self.discriminator([img_feats, real_ings], training=True)
            pred_fake = self.discriminator([img_feats, fake_ings], training=True)

            loss_real = self.bce(real_labels, pred_real)
            loss_fake = self.bce(fake_labels, pred_fake)
            disc_loss = (loss_real + loss_fake) / 2.0

        grads_disc = tape_disc.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(grads_disc, self.discriminator.trainable_variables))

        # 3) Train Generator (and optionally encoder)
        with tf.GradientTape() as tape_gen:
            noise = tf.random.normal((batch_size, NOISE_DIM))
            fake_ings = self.generator([noise, img_feats], training=True)

            pred_fake_for_gen = self.discriminator([img_feats, fake_ings], training=False)
            adv_loss = self.bce(real_labels, pred_fake_for_gen)

            l1_loss = tf.reduce_mean(tf.abs(real_ings - fake_ings))

            gen_loss = adv_loss + 10.0 * l1_loss

        gen_vars = self.generator.trainable_variables + self.encoder.trainable_variables
        grads_gen = tape_gen.gradient(gen_loss, gen_vars)
        self.gen_optimizer.apply_gradients(zip(grads_gen, gen_vars))

        return {"disc_loss": disc_loss, "gen_loss": gen_loss, "adv_loss": adv_loss, "l1_loss": l1_loss}


# ----------- 6. Supervised baseline (Appendix 1) -----------
def build_supervised_model(sup_encoder):
    """
    Same CNN encoder, trained directly with binary cross-entropy -
    no adversarial training. Used as a comparison point (see README).
    """
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    feat = sup_encoder(inputs)

    x = layers.Dense(256, activation="relu")(feat)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(TOP_K_INGREDIENTS, activation="sigmoid")(x)

    return models.Model(inputs, outputs, name="sup_ingredient_predictor")


# ----------- 7. Inference -----------
def predict_ingredients_for_image(image_path, image_encoder, generator, ingredient_vocab, threshold=0.5):
    im = Image.open(image_path).convert("RGB")
    im = im.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(im).astype("float32")
    arr = arr / 127.5 - 1.0
    img = tf.expand_dims(arr, axis=0)

    img_feat = image_encoder(img, training=False)
    noise = tf.random.normal((1, NOISE_DIM))
    fake_vec = generator([noise, img_feat], training=False).numpy()[0]

    indices = np.where(fake_vec >= threshold)[0]
    predicted_ingredients = [ingredient_vocab[i] for i in indices]
    return predicted_ingredients


if __name__ == "__main__":
    # Build the three networks
    image_encoder = build_image_encoder()
    generator = build_generator()
    discriminator = build_discriminator()

    gan = RecipeGAN(image_encoder, generator, discriminator)
    gan.compile(
        gen_optimizer=tf.keras.optimizers.Adam(1e-4),
        disc_optimizer=tf.keras.optimizers.Adam(1e-4),
    )

    # Example usage (requires your own df_model / ingredient_vectors):
    # gan_dataset = build_dataset(df_model, ingredient_vectors)
    # history = gan.fit(gan_dataset, epochs=20)
    #
    # predict_ingredients_for_image(
    #     "my_dish_photo.jpg", image_encoder, generator, ingredient_vocab
    # )
    image_encoder.summary()
    generator.summary()
    discriminator.summary()
