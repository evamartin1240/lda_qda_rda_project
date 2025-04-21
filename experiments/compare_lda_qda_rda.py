# experiments/compare_lda_qda_rda.py

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to the path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.generate_synthetic_data import generate_complex_dataset
from models.lda import LDA
from models.qda import QDA
from models.rda import RDA

from sklearn.metrics import accuracy_score

# 1. Define noise level
NOISE_LEVEL = 0.9  

# 2. Generate the dataset
X, y = generate_complex_dataset(n_samples_per_class=50, random_state=42, noise=NOISE_LEVEL)

# 3. Train the models
lda = LDA()
lda.fit(X, y)
y_pred_lda = lda.predict(X)

qda = QDA()
qda.fit(X, y)
y_pred_qda = qda.predict(X)

rda = RDA(alpha=0.5)
rda.fit(X, y)
y_pred_rda = rda.predict(X)

models = [lda, qda, rda]
titles = [
    f"LDA Decision Boundary (noise={NOISE_LEVEL})",
    f"QDA Decision Boundary (noise={NOISE_LEVEL})",
    f"RDA Decision Boundary (alpha=0.5, noise={NOISE_LEVEL})"
]
predictions = [y_pred_lda, y_pred_qda, y_pred_rda]

# 4. Compute accuracy
print(f"Classification accuracies (noise={NOISE_LEVEL}):")
for model_name, y_pred in zip(["LDA", "QDA", "RDA (alpha=0.5)"], predictions):
    acc = accuracy_score(y, y_pred)
    print(f"{model_name}: {acc:.4f}")

# 5. Plot decision boundaries
def plot_decision_boundary(model, X, y, title):
    """
    Plot the decision boundary of a classification model.
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.predict(grid)
    preds = preds.reshape(xx.shape)

    plt.contourf(xx, yy, preds, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title(title)
    plt.xlabel('X1')
    plt.ylabel('X2')

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, model, title in zip(axes, models, titles):
    plt.sca(ax)
    plot_decision_boundary(model, X, y, title)

plt.tight_layout()
plt.show()

