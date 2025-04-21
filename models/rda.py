# models/rda.py

import numpy as np

class RDA:
    def __init__(self, alpha=0.5):
        """
        Initialize the RDA model.

        Parameters:
        - alpha: Regularization parameter between 0 (LDA) and 1 (QDA)
        """
        self.alpha = alpha
        self.means_ = None
        self.pooled_covariance_ = None
        self.covariances_ = None
        self.priors_ = None
        self.classes_ = None
        self.regularized_covariances_ = None

    def fit(self, X, y):
        """
        Fit the RDA model according to the given training data.
        """
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        n_classes = len(self.classes_)

        self.means_ = np.zeros((n_classes, n_features))
        self.covariances_ = []
        self.priors_ = np.zeros(n_classes)
        self.pooled_covariance_ = np.zeros((n_features, n_features))

        # Compute means, class covariances, priors
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.means_[idx] = X_c.mean(axis=0)
            cov_c = np.cov(X_c, rowvar=False)
            self.covariances_.append(cov_c)
            self.pooled_covariance_ += cov_c * (X_c.shape[0] - 1)
            self.priors_[idx] = X_c.shape[0] / X.shape[0]

        self.pooled_covariance_ /= (X.shape[0] - n_classes)

        # Compute regularized covariances
        self.regularized_covariances_ = []
        for cov_k in self.covariances_:
            reg_cov = (1 - self.alpha) * self.pooled_covariance_ + self.alpha * cov_k
            self.regularized_covariances_.append(reg_cov)

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
        for mean, reg_cov, prior in zip(self.means_, self.regularized_covariances_, self.priors_):
            cov_inv = np.linalg.inv(reg_cov)
            term1 = -0.5 * np.log(np.linalg.det(reg_cov))
            term2 = -0.5 * (x - mean).T @ cov_inv @ (x - mean)
            term3 = np.log(prior)
            score = term1 + term2 + term3
            scores.append(score)
        return scores

