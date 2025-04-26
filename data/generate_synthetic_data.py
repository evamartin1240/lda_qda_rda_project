# data/generate_synthetic_data.py

import numpy as np
import os
import pandas as pd

def generate_simple_dataset(n_samples_per_class=50, random_state=None, noise=0.0):
    if random_state is not None:
        np.random.seed(random_state)

    X0 = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]], size=n_samples_per_class)
    y0 = np.zeros(n_samples_per_class)

    X1 = np.random.multivariate_normal(mean=[2, 2], cov=[[1, 0.5], [0.5, 1]], size=n_samples_per_class)
    y1 = np.ones(n_samples_per_class)

    X = np.vstack((X0, X1))
    y = np.hstack((y0, y1))

    if noise > 0.0:
        X += np.random.normal(0, noise, X.shape)

    return X, y

def generate_complex_dataset(n_samples_per_class=50, random_state=None, noise=0.0):
    if random_state is not None:
        np.random.seed(random_state)

    X0 = np.random.multivariate_normal(mean=[0, 0], cov=[[3, 0], [0, 1]], size=n_samples_per_class)
    y0 = np.zeros(n_samples_per_class)

    X1 = np.random.multivariate_normal(mean=[2, 2], cov=[[1, 0], [0, 3]], size=n_samples_per_class)
    y1 = np.ones(n_samples_per_class)

    X = np.vstack((X0, X1))
    y = np.hstack((y0, y1))

    if noise > 0.0:
        X += np.random.normal(0, noise, X.shape)

    return X, y

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Generate and save simple dataset
    X_simple, y_simple = generate_simple_dataset(n_samples_per_class=50, random_state=42, noise=0.5)
    df_simple = pd.DataFrame(X_simple, columns=["feature1", "feature2"])
    df_simple["label"] = y_simple.astype(int)
    print(f"Generated a simple dataset. Saved to: {script_dir}")
    df_simple.to_csv(os.path.join(script_dir, "simple_dataset.csv"), index=False)

    # Generate and save complex dataset
    X_complex, y_complex = generate_complex_dataset(n_samples_per_class=50, random_state=42, noise=0.5)
    df_complex = pd.DataFrame(X_complex, columns=["feature1", "feature2"])
    df_complex["label"] = y_complex.astype(int)
    print(f"Generated a complex dataset. Saved to: {script_dir}")
    df_complex.to_csv(os.path.join(script_dir, "complex_dataset.csv"), index=False)
