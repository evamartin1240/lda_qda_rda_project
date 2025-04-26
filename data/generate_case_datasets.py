# data/generate_case_datasets.py

import numpy as np

def generate_dataset_case(case="lda_friendly", n_samples_per_class=50, noise=0.3, random_state=42):
    if random_state is not None:
        np.random.seed(random_state)

    if case == "lda_friendly":
        mean0 = [0, 0]
        mean1 = [2, 2]
        cov = [[1, 0.5], [0.5, 1]]
        X0 = np.random.multivariate_normal(mean0, cov, n_samples_per_class)
        X1 = np.random.multivariate_normal(mean1, cov, n_samples_per_class)

    elif case == "qda_friendly":
        mean0 = [0, 0]; cov0 = [[3, 1], [1, 2]]
        mean1 = [3, 3]; cov1 = [[1, -0.8], [-0.8, 1.5]]
        X0 = np.random.multivariate_normal(mean0, cov0, n_samples_per_class)
        X1 = np.random.multivariate_normal(mean1, cov1, n_samples_per_class)

    elif case == "rda_friendly":
        mean0 = [0, 0]; cov0 = [[3, 1], [1, 2]]
        mean1 = [3, 3]; cov1 = [[1, -0.8], [-0.8, 1.5]]
        X0 = np.random.multivariate_normal(mean0, cov0, n_samples_per_class)
        X1 = np.random.multivariate_normal(mean1, cov1, n_samples_per_class)

        # add ambiguous boundary points
        mid = [(mean0[0]+mean1[0])/2, (mean0[1]+mean1[1])/2]
        boundary = np.random.multivariate_normal(mid, [[1, 0], [0, 1]], int(0.2*n_samples_per_class))
        X_boundary = boundary
        y_boundary = np.random.randint(0, 2, len(X_boundary))
    
    elif case == "circular":
        theta = np.random.uniform(0, 2*np.pi, n_samples_per_class)
        r0 = np.random.normal(1, 0.2, n_samples_per_class)
        r1 = np.random.normal(3, 0.2, n_samples_per_class)
        X0 = np.c_[r0 * np.cos(theta), r0 * np.sin(theta)]
        X1 = np.c_[r1 * np.cos(theta), r1 * np.sin(theta)]

    elif case == "unbalanced":
        mean0 = [0, 0]; cov0 = [[2, 0.5], [0.5, 1]]
        mean1 = [2, 2]; cov1 = [[1, 0], [0, 2]]
        X0 = np.random.multivariate_normal(mean0, cov0, n_samples_per_class * 2)
        X1 = np.random.multivariate_normal(mean1, cov1, n_samples_per_class)

    else:
        raise ValueError(f"Unknown case: {case}")

    if case == "rda_friendly":
        X = np.vstack((X0, X1, X_boundary))
        y = np.hstack((np.zeros(len(X0)), np.ones(len(X1)), y_boundary))
    elif case == "unbalanced":
        X = np.vstack((X0, X1))
        y = np.hstack((np.zeros(len(X0)), np.ones(len(X1))))
    else:
        X = np.vstack((X0, X1))
        y = np.hstack((np.zeros(n_samples_per_class), np.ones(n_samples_per_class)))

    if noise > 0:
        X += np.random.normal(0, noise, X.shape)

    return X, y

if __name__ == "__main__":
    import os
    import pandas as pd

    output_dir = os.path.join(os.path.dirname(__file__), "case_datasets")
    os.makedirs(output_dir, exist_ok=True)

    cases = ["lda_friendly", "qda_friendly", "rda_friendly", "circular", "unbalanced"]

    for case in cases:
        X, y = generate_dataset_case(case=case, n_samples_per_class=60, noise=0.4, random_state=42)
        df = pd.DataFrame(X, columns=["feature1", "feature2"])
        df["label"] = y.astype(int)
        csv_path = os.path.join(output_dir, f"{case}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved {case} dataset to {csv_path}")

