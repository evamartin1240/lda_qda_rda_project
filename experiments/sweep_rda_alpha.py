# experiments/sweep_rda_alpha.py

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to the path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.generate_synthetic_data import generate_complex_dataset
from models.rda import RDA

# 1. Define noise level
NOISE_LEVEL = 0.3  # Set 0.0 for no noise, or e.g., 0.3 for moderate noise

# 2. Generate the dataset
X, y = generate_complex_dataset(n_samples_per_class=50, random_state=42, noise=NOISE_LEVEL)

# 3. Define alpha values to sweep
alpha_values = np.linspace(0.0, 1.0, 6)  # Six values: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0

# 4. Train RDA models for each alpha
models = []

for alpha in alpha_values:
    model = RDA(alpha=alpha)
    model.fit(X, y)
    models.append((model, alpha))

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
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for ax, (model, alpha) in zip(axes.flatten(), models):
    plt.sca(ax)
    plot_decision_boundary(model, X, y, f"RDA (alpha={alpha:.1f}, noise={NOISE_LEVEL})")

plt.tight_layout()
plt.show()

