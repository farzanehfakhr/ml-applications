"""
Image-to-Recipe Retrieval: given a photo of a dish, find the most visually
similar recipes in a dataset of 13,000+ recipe images.

Instead of classifying an image into a fixed label, a small custom CNN is
used purely as an embedding generator: each image is mapped to a 256-D,
L2-normalized vector, and retrieval is done with k-nearest-neighbors over
those embeddings (cosine/Euclidean similarity on the unit sphere).

Pipeline:
  1. build_small_cnn_embedding - the CNN architecture (3 conv/pool blocks ->
     global average pooling -> 256-D dense embedding -> L2 normalize)
  2. load_img_to_tensor / batched - preprocess and batch images for inference
  3. embed the entire dataset (13,471 images) with the CNN
  4. index the embeddings with sklearn's NearestNeighbors
  5. query_image_v2 - given a new photo, retrieve the top-K most similar recipes
  6. recall_at_k_generic / self_retrieval_rate - evaluate retrieval quality

Reconstructed from the author's own notebook screenshots (MET CS 767,
Assignment 3, Section 2: "Your CNN Application"). Transcribed, not re-derived.
The dataset itself (13,471 recipe images + a metadata dataframe `df` with
columns like `img_path`, `Title`) is not included here; swap in your own.
"""

import numpy as np
from tensorflow.keras import models, layers
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
from sklearn.neighbors import NearestNeighbors


IMG_SIZE = (224, 224)


# ----------- 1. CNN embedding model -----------
def build_small_cnn_embedding(input_shape=(224, 224, 3), embed_dim=256):
    """
    A small, transparent CNN used purely as a feature generator:
    3 conv/pool blocks -> global average pooling -> dense(embed_dim)
    -> L2 normalization, so outputs are unit-length vectors suitable for
    k-NN / cosine similarity search.
    """
    model = models.Sequential(name="small_cnn_embedding")
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(embed_dim))
    model.add(layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1), name="l2_norm"))
    return model


small_cnn = build_small_cnn_embedding()
small_cnn.summary()


# ----------- 2. Image loading / batching -----------
def load_img_to_tensor(path):
    img = kimage.load_img(path, target_size=IMG_SIZE)
    arr = kimage.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return arr


def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield i, iterable[i:i + n]


# ----------- 3. Embed the whole dataset -----------
def embed_dataset(df, batch_size=32):
    """
    df: a dataframe with an "img_path" column pointing at recipe images.
    Returns an (N, 256) float32 array of L2-normalized embeddings, in the
    same row order as df.
    """
    paths_full = df["img_path"].tolist()
    print("Total images (full dataset):", len(paths_full))

    all_vecs = []
    for i, batch_paths in batched(paths_full, batch_size):
        batch_arr = np.vstack([load_img_to_tensor(p) for p in batch_paths])
        vecs = small_cnn.predict(batch_arr, verbose=0)
        all_vecs.append(vecs)
        if (i // batch_size) % 10 == 0:
            print(f"processed up to index {i + len(batch_paths)} / {len(paths_full)}")

    embeddings = np.vstack(all_vecs).astype("float32")
    print("Embeddings shape:", embeddings.shape)
    return embeddings


# ----------- 4. Build the k-NN index -----------
def build_knn_index(embeddings, n_neighbors=10):
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(embeddings)
    return knn


# ----------- 5. Query: given a photo, retrieve the top-K most similar recipes -----------
def query_image_v2(img_path, df, knn, topk=10):
    arr = load_img_to_tensor(img_path)
    qvec = small_cnn.predict(arr, verbose=0)

    dists, idxs = knn.kneighbors(qvec, n_neighbors=topk, return_distance=True)
    dists = dists[0].tolist()
    idxs = idxs[0].tolist()

    out = df.iloc[idxs].copy()
    out["distance"] = dists
    # embeddings are L2-normalized, so squared Euclidean distance maps
    # cleanly onto a 0-1 similarity score
    out["similarity"] = (2 - np.array(dists)) / 2.0
    return out.reset_index(drop=True)


# ----------- 6. Evaluation -----------
def recall_at_k_generic(df, query_fn, sample_n=100, k=10, seed=0):
    """
    A retrieved recipe is counted as correct only if it has a matching
    normalized title key or at least three overlapping cleaned ingredients
    (strict proxy metric - see write-up for discussion of why this
    understates true visual-similarity quality).
    """
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(df), size=min(sample_n, len(df)), replace=False)
    hits = 0
    for i in idxs:
        q = df.iloc[i]
        res = query_fn(q["img_path"], topk=k)
        # NOTE: plug in your own title/ingredient-overlap match condition here
        if q["Title"] in res["Title"].values:
            hits += 1
    return hits / len(idxs)


def self_retrieval_rate(df_used, query_fn, sample_n=100):
    """
    Sanity check: querying with an image that is itself in the dataset
    should retrieve itself as the #1 result almost all the time.
    """
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(df_used), size=min(sample_n, len(df_used)), replace=False)
    hits = 0
    for i in idxs:
        q = df_used.iloc[i]
        res = query_fn(q["img_path"], topk=1)
        if res["img_path"].iloc[0] == q["img_path"]:
            hits += 1
    return hits / len(idxs)


if __name__ == "__main__":
    # Example usage (requires a dataframe `df` with an "img_path" and
    # "Title" column pointing at your own recipe image dataset):
    #
    # embeddings = embed_dataset(df)
    # knn = build_knn_index(embeddings, n_neighbors=10)
    # results = query_image_v2("my_dish_photo.jpg", df, knn, topk=10)
    # print(results[["Title", "similarity"]])
    #
    # On the author's 13,471-image recipe dataset:
    #   V2 (small CNN) Recall@10: 0.070
    #   V2 (small CNN) Recall@1:  0.010
    #   V2 self-retrieval (sanity check): 0.98
    pass
