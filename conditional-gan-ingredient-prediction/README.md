# Conditional GAN: Image-to-Ingredients Prediction

A conditional GAN ("RecipeGAN") that takes a photo of a dish and generates a plausible list of ingredients — and an honest account of where it worked and where it broke. Built for MET CS 767 (Boston University), Assignment 4.

> Code here was transcribed from screenshots in the original assignment write-up (the live Colab notebook was on a university account that's since been disabled).

## The application

**Goal.** Given a food photo, predict a multi-label set of likely ingredients — not a single class label, but a ranked, multi-hot vector over a ~200-word ingredient vocabulary.

**Data.** [Food Ingredients and Recipes Dataset with Images](https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images) (Kaggle) — ~14,000 food images, each paired with a real ingredient list.

**Architecture.** Three components, all conditioned on the input image:

```
Image Encoder (CNN)              Generator                    Discriminator
128×128×3 image                  noise (64-D) ⊕ img features   img features ⊕ ingredient vector
→ 4× Conv2D (strides 2)          → Dense(256) → Dense(256)     → Dense(256) → LeakyReLU
→ Flatten → Dense(256)           → Dense(200, sigmoid)         → Dense(256) → LeakyReLU
= 256-D image feature vector     = multi-hot ingredient probs  → Dense(1, sigmoid) real/fake
```

Training alternates: the discriminator sees (image, real ingredients) vs. (image, generator's ingredients) and learns to tell them apart; the generator is updated against both an adversarial loss (fool the discriminator) and an L1 loss against the true ingredient vector, so it's pushed toward both "looks real" and "is actually close."

## Results

Trained for 20 epochs on a subset of the dataset (`disc_loss` and `gen_loss` logged every epoch). A supervised baseline — the same image encoder, trained directly with binary cross-entropy against the true ingredient vectors, no adversarial training — was built for comparison (`build_supervised_model`).

**The GAN learned something, but not what was intended.** Early in training it produced realistic-sounding ingredient phrases ("1 tablespoon fresh lemon juice", "2 garlic cloves, finely chopped"), which is a positive sign the model picked up the general vocabulary of recipe text. But the predictions weren't meaningfully tied to the input image. A representative example from evaluation — three completely different dishes (a roast chicken dish, salt-and-pepper potatoes, and mac and cheese) all got nearly the same top prediction:

```
=== Supervised Example 1 === (miso-butter-roast-chicken-acorn-squash-panzanella.jpg)
Predicted (top-10): 2 large eggs (0.03), 2 tbsp vegetable oil (0.02), 1 large egg (0.02), ...

=== Supervised Example 2 === (crispy-salt-and-pepper-potatoes-dan-kluger.jpg)
Predicted (top-10): 2 large eggs (0.04), 2 tbsp vegetable oil (0.03), 1 large egg (0.03), ...

=== Supervised Example 3 === (thanksgiving-mac-and-cheese-erick-williams.jpg)
Predicted (top-10): 2 large eggs (0.02), 1 large egg (0.02), 2 tbsp vegetable oil (0.02), ...
```

**Why:** the ingredient vocabulary is heavily imbalanced — common items (eggs, oil, salt, sugar) appear in thousands of recipes, rare ones in only a handful. To minimize loss, both the GAN and the supervised model learned to default to the most frequent ingredients almost regardless of the image — a classic failure mode for high-dimensional, sparse multi-label prediction. The GAN additionally showed mode collapse (repeating a "safe" ingredient set that reliably fooled the discriminator) and, after more training, the opposite problem — overly long, noisy 20-30 item lists mixing baking and savory ingredients indiscriminately.

**GAN vs. supervised baseline:** the supervised model, trained with a direct per-ingredient loss, produced more consistent and image-appropriate ingredient sets (e.g. emphasizing grains/legumes/cheese for a salad-type image) than the adversarially-trained GAN, which optimizes only a global real/fake signal. The GAN is the more interesting generative model; the supervised baseline is the more *reliable* predictor — and that gap is itself the main finding of this project.

## Base DCGAN tuning exercise (MNIST, Section 1)

Before building the application above, the assignment started with a guided DCGAN-tuning exercise on MNIST digits (training set reduced to 20,000 images + a 5,000-image validation split, to make architecture/hyperparameter changes more visible):

| Change | Effect |
|---|---|
| Baseline DCGAN | Blurry but mostly identifiable digits |
| Reduced training data (60k → 20k) | Quality dropped sharply — discriminator overpowers a small generator |
| + extra Conv2DTranspose/BatchNorm/LeakyReLU block in generator (14×14) | Slightly smoother curves, moderate improvement |
| + extra conv refinement block in discriminator | Sharper digits, but inconsistent — some over-refined/distorted |
| + tuned Adam (gen LR 2e-4, disc LR 1e-4, β₁=0.5 for both) | **Clearest improvement** — more stable training, cleaner digit shapes |

The optimizer tuning (matching the hyperparameters recommended in the original DCGAN paper) produced a bigger improvement than either architecture change — a useful reminder that GAN training is often more sensitive to optimization dynamics than to model capacity.

## Files

| File | Description |
|---|---|
| `recipe_gan.py` | Image encoder, generator, discriminator, the `RecipeGAN` training loop, the supervised baseline, and inference |

## Tools

Python, TensorFlow/Keras, conditional GANs, multi-label prediction

---
*Completed as part of graduate coursework (METCS767, Machine Learning) at Boston University.*
