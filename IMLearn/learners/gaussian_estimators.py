from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """
    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def __str__(self):
        return f"({self.mu_}, {self.var_})\n"

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        m = X.size
        sum = X.sum()
        mu = sum / m
        if self.biased_:
            self.mu_ = mu - 1 / m
            self.var_ = np.sum(np.square(X - mu)) / m
        else:
            self.mu_ = mu
            self.var_ = np.sum(np.square(X - mu)) / (m - 1)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        denominator = np.sqrt(2 * np.pi * self.var_)
        exp = np.exp((-np.square(X - self.mu_)) / (2 * self.var_))
        return exp / denominator


    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        m = X.shape[0]
        return -0.5 * m * np.log(2 * np.pi * sigma) - (1 / (2 * sigma)) * np.sum(np.square(X - mu))


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def __str__(self):
        return f"{self.mu_} \n\n {self.cov_}\n"

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        ones_vector = np.ones(shape=(1, X.shape[0]))
        self.mu_ = (np.dot(ones_vector, X) / X.shape[0]).reshape(-1)
        m = X.shape[0]
        centered_matrix = X - self.mu_[None, :]
        centered_matrix_trans = centered_matrix.transpose()
        self.cov_ = centered_matrix_trans.dot(centered_matrix) / (m - 1)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        sigma_det = np.linalg.det(self.cov_)
        sigma_inv = np.linalg.inv(self.cov_)
        d = X.shape[0]
        X_minus_mu = X - self.mu_
        X_minus_mu_trans = X_minus_mu.transpose()
        denominator = np.sqrt(np.power(2 * np.pi, d) * sigma_det)
        exp = np.exp(-0.5 * (X_minus_mu_trans).dot(sigma_inv).dot(X_minus_mu))
        return exp / denominator

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        d = mu.size
        m = X.shape[0]
        sig_det = np.linalg.det(cov)
        sig_inv = np.linalg.inv(cov)
        X_minus_mu = X - mu[None, :]
        X_minus_mu_transpose = X_minus_mu.transpose()
        return ((-d * m) * 0.5) * np.log(2 * np.pi) - (m * 0.5) * np.log(sig_det) - (0.5) * np.trace((X_minus_mu.dot(sig_inv)).dot(X_minus_mu_transpose))
