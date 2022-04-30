from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # calculate classes and pi
        self.classes_, counts = np.unique(y, return_counts=True)
        self.pi_ = counts / y.shape[0]

        # calculate mu-es
        mu_list = []
        for i, classifier in enumerate(self.classes_):
            mk = counts[i]
            class_appear = np.where(y == classifier)[0]
            row = np.sum(X[class_appear], axis=0) / mk
            mu_list.append(row)
        self.mu_ = np.array(mu_list)

        # calculate cov & cov_inv
        m = X.shape[0]
        d = X.shape[1]
        k = self.classes_.size
        cov = np.zeros((d, d))
        for i, classifier in enumerate(self.classes_):
            class_appear = np.where(y == classifier)[0]
            class_mat = X[class_appear] - self.mu_[i]
            cov = cov + (class_mat.T @ class_mat)
        self.cov_ = cov / (m - k)
        self._cov_inv = np.linalg.inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        pi_likelihood = self.likelihood(X)
        return self.classes_[np.argmax(pi_likelihood * self.pi_, axis=1)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        likelihood_list = []
        d = X.shape[1]
        sigma_det = np.linalg.det(self.cov_)
        denominator = np.sqrt(np.power(2 * np.pi, d) * sigma_det)
        for mu in self.mu_:
            X_minus_mu = X - mu
            X_minus_mu_trans = X_minus_mu.T
            exp = np.exp(-0.5 * X_minus_mu @ self._cov_inv @ X_minus_mu_trans)
            likelihood = np.diag(exp) / denominator
            likelihood_list.append(likelihood)
        return np.array(likelihood_list).T

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        pred_y = self._predict(X)
        return misclassification_error(y, pred_y)
