# data/generate_synthetic_data.py

import numpy as np

def generate_simple_dataset(n_samples_per_class=50, random_state=None, noise=0.0):
    """
    Generate a simple dataset where classes have the same covariance.

    Parameters:
    - n_samples_per_class: Number of samples per class.
    - random_state: Seed for reproducibility.
    - noise: Standard deviation of Gaussian noise to add to the features.
    
    Returns:
    - X: Feature matrix.
    - y: Labels.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Class 0: centered at (0, 0)
    X0 = np.random.multivariate_normal(
        mean=[0, 0],
        cov=[[1, 0.5], [0.5, 1]],
        size=n_samples_per_class
    )
    y0 = np.zeros(n_samples_per_class)

    # Class 1: centered at (2, 2)
    X1 = np.random.multivariate_normal(
        mean=[2, 2],
        cov=[[1, 0.5], [0.5, 1]],
        size=n_samples_per_class
    )
    y1 = np.ones(n_samples_per_class)

    X = np.vstack((X0, X1))
    y = np.hstack((y0, y1))

    # Add Gaussian noise if specified
    if noise > 0.0:
        X += np.random.normal(0, noise, X.shape)

    return X, y

def generate_complex_dataset(n_samples_per_class=50, random_state=None, noise=0.0):
    """
    Generate a more complex dataset where classes have different covariances.

    Parameters:
    - n_samples_per_class: Number of samples per class.
    - random_state: Seed for reproducibility.
    - noise: Standard deviation of Gaussian noise to add to the features.
    
    Returns:
    - X: Feature matrix.
    - y: Labels.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Class 0: elongated along the x-axis
    X0 = np.random.multivariate_normal(
        mean=[0, 0],
        cov=[[3, 0], [0, 1]],
        size=n_samples_per_class
    )
    y0 = np.zeros(n_samples_per_class)

    # Class 1: elongated along the y-axis
    X1 = np.random.multivariate_normal(
        mean=[4, 4],
        cov=[[1, 0], [0, 3]],
        size=n_samples_per_class
    )
    y1 = np.ones(n_samples_per_class)

    X = np.vstack((X0, X1))
    y = np.hstack((y0, y1))

    # Add Gaussian noise if specified
    if noise > 0.0:
        X += np.random.normal(0, noise, X.shape)

    return X, y

