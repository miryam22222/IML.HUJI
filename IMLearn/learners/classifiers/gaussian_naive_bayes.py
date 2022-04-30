from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

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

        # calculate vars
        vars = []
        for i, classifier in enumerate(self.classes_):
            class_appear = np.where(y == classifier)[0]
            row = X[class_appear].var(axis=0, ddof=1)
            vars.append(row)
        self.vars_ = np.array(vars)

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
        for i, classifier in enumerate(self.classes_):
            mu = self.mu_[i]
            var = self.vars_[i]
            X_minus_mu = X - mu
            denominator = np.sqrt(var * 2 * np.pi)
            exp = np.exp(-(np.square(X_minus_mu) / (2 * var)))
            likelihood = np.product(exp / denominator, axis=1)
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
