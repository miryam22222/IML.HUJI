import numpy as np
from ..base import BaseEstimator
from typing import Callable, NoReturn
from ..metrics.loss_functions import misclassification_error


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        m = X.shape[0]
        self.weights = []
        self.models_ = []
        self.D_ = np.full((m, ), 1 / m)
        for i in range(self.iterations_):
            h = self.wl_().fit(X, y * self.D_)
            self.models_.append(h)
            y_pred = h.predict(X)
            eps = np.sum(self.D_ * (y != y_pred))
            weight = 0.5 * np.log(1 / eps - 1)
            self.weights.append(weight)
            self.D_ = self.D_ * np.exp(-y * weight * y_pred)
            self.D_ = self.D_ / sum(self.D_)

    def _predict(self, X):
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
        return self.partial_predict(X, len(self.models_))

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
        return self.partial_loss(X, y, len(self.models_))

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        # pred_sum = np.zeros(X.shape[0])
        # for i, model in enumerate(self.models_):
        #     pred = model.predict(X)
        #     w = self.weights[i]
        #     pred_sum += (model.predict(X) * self.weights[i])
        # x = np.sum(pred_sum)
        # return np.sign(pred_sum)
        pred = np.sign(np.sum([self.models_[t].predict(X) * self.weights[t] for t in range(T)], axis=0))
        return pred

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self.partial_predict(X, T)
        return misclassification_error(y, y_pred)
