# experiments/test_lda.py

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to the path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.lda import LDA

# 1. Create a simple synthetic dataset
np.random.seed(0)
n_samples = 50

# Class 0: centered at (0, 0)
X0 = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]], size=n_samples)
y0 = np.zeros(n_samples)

# Class 1: centered at (2, 2)
X1 = np.random.multivariate_normal(mean=[2, 2], cov=[[1, -0.5], [-0.5, 1]], size=n_samples)
y1 = np.ones(n_samples)

# Combine the two classes
X = np.vstack((X0, X1))
y = np.hstack((y0, y1))

# 2. Train the LDA model
lda = LDA()
lda.fit(X, y)
y_pred = lda.predict(X)

# 3. Plot the decision boundary
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
    plt.show()

plot_decision_boundary(lda, X, y, "LDA Decision Boundary")

