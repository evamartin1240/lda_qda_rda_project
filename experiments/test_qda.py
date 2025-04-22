# experiments/test_qda.py

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd

# Add project root to the path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.qda import QDA

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

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} path/to/dataset.csv")
        sys.exit(1)

    dataset_path = sys.argv[1]

    if not os.path.exists(dataset_path):
        print(f"Error: File '{dataset_path}' does not exist.")
        sys.exit(1)

    # Load dataset
    df = pd.read_csv(dataset_path)
    if not all(col in df.columns for col in ["feature1", "feature2", "label"]):
        print("Error: CSV must contain 'feature1', 'feature2', and 'label' columns.")
        sys.exit(1)

    X = df[["feature1", "feature2"]].values
    y = df["label"].values

    # Train the QDA model
    qda = QDA()
    qda.fit(X, y)

    # Plot
    plot_decision_boundary(qda, X, y, "QDA Decision Boundary")

if __name__ == "__main__":
    main()
