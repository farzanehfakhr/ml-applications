# Coursework Fundamentals (Assignments 1–2)

The first two assignments in this course were shorter, guided exercises — practice with a given technique before the later, fully original applications (which have their own folders in this repo). Summarized here rather than given full code, since they were closer to tutorials than standalone projects.

## Assignment 1: Decision Tree Classifier

Built a decision tree to classify students as "High," "Moderate," or "Needs Support" from the `StudentsPerformance.csv` dataset, using features like gender, lunch type, test-preparation course completion, and engineered metrics (average score, score spread across subjects). Used `criterion="entropy"` (information gain) to balance accuracy with interpretability — the goal was an educational tool where a teacher could see *why* a student was flagged, not just the label. Test accuracy: ~79%. A follow-up experiment retrained the tree on a deliberately inconsistent version of the dataset, which showed the model's predictions could shift for borderline students even under small data perturbations — a useful lesson in decision tree sensitivity.

## Assignment 2: Neural Networks Intro

Two parts: a guided hyperparameter-tuning warm-up, and a more substantial original application.

**Warm-up:** standard MNIST digit classification, with the training set deliberately reduced to 10% (baseline accuracy ~97–98% on the full set leaves no room to show improvement) to make the effect of architecture/hyperparameter changes visible — tested network width, an added hidden layer, L2 regularization, activation function, and dropout.

**Application — student learning persona classifier:** a two-stage model combining unsupervised and supervised learning. K-Means clustering first groups students into three behavioral personas ("Highly Structured," "Moderately Engaged," "At-Risk Independent") based on study hours, attendance, parental support, and prior grades; a feed-forward neural network then learns to classify students into these discovered personas directly from the raw features. Architecture tuning found a tapered multi-layer network avoided both underfitting (too shallow) and overfitting; Adam with learning rate 0.003 outperformed SGD/RMSprop, and early stopping (patience=10, monitoring validation loss) typically halted training between 70–90 epochs. Validation accuracy: ~88–90%.

*Note: this second part is a genuine original application in its own right (it's what's referenced on the resume as the "NN + K-Means clustering" project) — it was grouped into this fundamentals note mainly to keep this repo's scope manageable, not because it's less substantial than the other four. Let me know if you'd like it pulled out into its own full folder with transcribed code, the same way the other four were done.*

---
*Completed as part of graduate coursework (METCS767, Machine Learning) at Boston University.*
