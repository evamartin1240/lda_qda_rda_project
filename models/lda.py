# models/lda.py

import numpy as np

class LDA:
    def __init__(self):
        self.means_ = None
        self.covariance_ = None
        self.priors_ = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Fit the LDA model according to the given training data.
        """
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        n_classes = len(self.classes_)

        self.means_ = np.zeros((n_classes, n_features))
        self.covariance_ = np.zeros((n_features, n_features))
        self.priors_ = np.zeros(n_classes)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.means_[idx] = X_c.mean(axis=0)
            self.covariance_ += np.cov(X_c, rowvar=False) * (X_c.shape[0] - 1)
            self.priors_[idx] = X_c.shape[0] / X.shape[0]

        self.covariance_ /= (X.shape[0] - n_classes)  # Pooled covariance estimate

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.
        """
        discriminants = np.array([self._discriminant_function(x) for x in X])
        return self.classes_[np.argmax(discriminants, axis=1)]

    def _discriminant_function(self, x):
        """
        Compute the discriminant scores for a single sample x.
        """
        scores = []
        cov_inv = np.linalg.inv(self.covariance_)
        for mean, prior in zip(self.means_, self.priors_):
            score = x @ cov_inv @ mean - 0.5 * mean.T @ cov_inv @ mean + np.log(prior)
            scores.append(score)
        return scores

