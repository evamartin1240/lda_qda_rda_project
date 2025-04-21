# experiments/test_rda.py

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to the path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.rda import RDA

# 1. Create a synthetic dataset with different covariances
np.random.seed(42)
n_samples = 50

# Class 0: centered at (0, 0), elongated along x-axis
X0 = np.random.multivariate_normal(
    mean=[0, 0],
    cov=[[3, 0], [0, 1]],
    size=n_samples
)
y0 = np.zeros(n_samples)

# Class 1: centered at (4, 4), elongated along y-axis
X1 = np.random.multivariate_normal(
    mean=[4, 4],
    cov=[[1, 0], [0, 3]],
    size=n_samples
)
y1 = np.ones(n_samples)

# Combine the two classes
X = np.vstack((X0, X1))
y = np.hstack((y0, y1))

# 2. Train RDA models with different alpha values
alpha_values = [0.0, 0.5, 1.0]
models = []

for alpha in alpha_values:
    model = RDA(alpha=alpha)
    model.fit(X, y)
    models.append(model)

# 3. Plot the decision boundaries
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

for ax, model, alpha in zip(axes, models, alpha_values):
    plt.sca(ax)
    plot_decision_boundary(model, X, y, f"RDA Decision Boundary (alpha={alpha})")

plt.tight_layout()
plt.show()

