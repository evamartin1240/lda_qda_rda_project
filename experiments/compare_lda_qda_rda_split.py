# experiments/compare_lda_qda_rda_split_csv.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.lda import LDA
from models.qda import QDA
from models.rda import RDA

# Argument parser
parser = argparse.ArgumentParser(description="Compare LDA, QDA and RDA with train/test split using a CSV file")
parser.add_argument('csv', type=str, help="Path to CSV file")
args = parser.parse_args()

# Load dataset
df = pd.read_csv(args.csv)

X = df[["feature1", "feature2"]].values
y = df["label"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Initialize models
lda = LDA()
qda = QDA()
rda = RDA(alpha=0.5)

models = [lda, qda, rda]
model_names = ["LDA", "QDA", "RDA (alpha=0.5)"]

# Train
for model in models:
    model.fit(X_train, y_train)

# Evaluate
print(f"Classification accuracies on test set (CSV = {args.csv}):")
for name, model in zip(model_names, models):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name}: {acc:.4f}")

# Plot decision boundary on train data
def plot_decision_boundary(model, X, y, title):
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

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, model, name in zip(axes, models, model_names):
    plt.sca(ax)
    plot_decision_boundary(model, X_train, y_train, f"{name} (train data)")

plt.tight_layout()
plt.show()
