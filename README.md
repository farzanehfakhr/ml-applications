# Machine Learning Applications

Six hands-on machine learning projects completed for METCS767 (Boston University), spanning classical ML, deep learning, generative models, evolutionary algorithms, and probabilistic reasoning. Each original Colab notebook lived on a university account that's since been deactivated, so the code in the four main folders below was carefully transcribed back from the screenshots embedded in the original write-ups — not rewritten from scratch. Where possible, the transcriptions were verified by running them and checking the output against the documented results (see each folder's README for specifics).

## Projects

**[CNN Image-to-Recipe Retrieval](cnn-image-recipe-retrieval/)**
A custom CNN used as an embedding generator (not a classifier) to retrieve visually similar recipes for a photo of a dish, searched across 13,471 images via k-nearest-neighbors. Also includes the preceding CIFAR-10 architecture/hyperparameter ablation study.

**[Conditional GAN: Image-to-Ingredients Prediction](conditional-gan-ingredient-prediction/)**
A conditional GAN ("RecipeGAN") that predicts a dish's likely ingredients from a photo — plus an honest account of where it broke down (mode collapse, ingredient-vocabulary imbalance) and how a simpler supervised baseline ended up outperforming it.

**[Genetic Algorithm: TravOrder (TSP)](genetic-algorithm-travorder/)**
A genetic algorithm solving the Traveling Salesman Problem with a custom "k-th nearest unvisited city" chromosome encoding that keeps every individual automatically valid. Verified: reproduces the original notebook's exact results.

**[Bayesian Network: Allergy Risk Assessment](bayesian-network-allergy-risk/)**
A Bayesian network estimating severe-allergic-reaction risk for a student on a school trip, with full conditional probability tables, worked examples, and a sensitivity analysis. Verified: matches the original worked examples to 4+ decimal places.

**[Coursework Fundamentals](coursework-fundamentals.md)** (Assignments 1–2, summarized rather than fully reconstructed)
A decision tree classifier for student performance (~79% accuracy) and a neural-network + K-Means student-persona classifier (~88–90% validation accuracy).

## Files

| Path | Description |
|---|---|
| `cnn-image-recipe-retrieval/` | CNN embedding model + retrieval pipeline, CIFAR-10 experiments |
| `conditional-gan-ingredient-prediction/` | RecipeGAN: encoder, generator, discriminator, training loop |
| `genetic-algorithm-travorder/` | TravOrder GA for the Traveling Salesman Problem |
| `bayesian-network-allergy-risk/` | Allergy-risk Bayesian network (pgmpy + dependency-free verification) |
| `coursework-fundamentals.md` | Decision tree + neural net / K-Means persona classifier, summarized |
| `Course_Presentation.pptx` | Course presentation summarizing the work |

## Tools

Python, TensorFlow/Keras, scikit-learn, pgmpy, Neural Networks, CNNs, GANs, Genetic Algorithms, Bayesian Networks

---
*Completed as part of graduate coursework (METCS767, Machine Learning) at Boston University.*
