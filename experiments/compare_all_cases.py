# experiments/compare_all_cases.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.lda import LDA
from models.qda import QDA
from models.rda import RDA


def plot_decision_boundary(model, X, y, X_test, y_test, title, ax):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.predict(grid).reshape(xx.shape)

    ax.contourf(xx, yy, preds, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', label='Train')

    # Overlay test points, color coded by correctness
    y_pred_test = model.predict(X_test)
    correct = y_pred_test == y_test
    ax.scatter(X_test[correct, 0], X_test[correct, 1], marker='o', facecolors='none', edgecolors='g', label='Test correct')
    ax.scatter(X_test[~correct, 0], X_test[~correct, 1], marker='x', c='r', label='Test wrong')

    ax.set_title(title)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.legend(fontsize='small')


def evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = [LDA(), QDA(), RDA(alpha=0.5)]
    names = ["LDA", "QDA", "RDA (alpha=0.5)"]
    accs = []

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for model, name, ax in zip(models, names, axes):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accs.append(acc)
        plot_decision_boundary(model, X_train, y_train, X_test, y_test, f"{name}\nAcc: {acc:.2f}", ax)

    plt.tight_layout()
    return accs, fig


if __name__ == "__main__":
    dataset_dir = "data/case_datasets"
    cases = ["lda_friendly", "qda_friendly", "rda_friendly", "circular", "unbalanced"]

    summary = []
    os.makedirs("results/plots", exist_ok=True)

    for case in cases:
        path = os.path.join(dataset_dir, f"{case}.csv")
        df = pd.read_csv(path)
        X = df[["feature1", "feature2"]].values
        y = df["label"].values

        accs, fig = evaluate_models(X, y)
        summary.append([case] + accs)

        fig.savefig(f"results/plots/{case}.png", dpi=300, bbox_inches="tight")
        print(f"Saved plot for {case}")

    # Print summary table
    print("\nSummary Table:")
    print(f"{'Dataset':<15} {'LDA':<8} {'QDA':<8} {'RDA':<8}")
    for row in summary:
        print(f"{row[0]:<15} {row[1]:<8.4f} {row[2]:<8.4f} {row[3]:<8.4f}")

