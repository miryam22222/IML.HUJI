import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
import plotly.express as px


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        data = load_dataset('../datasets/' + f)
        X = data[0]
        y = data[1]

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def callback(fit: Perceptron, curr_x: np.ndarray, curr_y: int):
            losses.append(fit._loss(X, y))

        perceptron = Perceptron(callback=callback)
        perceptron.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        fig = px.line(y=losses).update_xaxes(title='Iterations').update_yaxes(title='Misclassification Loss')
        fig.update_layout(title=f'Misclassification Loss in {n} Samples as a Function of Perceptron Algorithm',
                          title_x=0.5, title_font_size=20, width=1000)
        fig.write_image(f'../ANSWERS/ex3/{n}.png')


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        data = load_dataset('../datasets/' + f)
        X = data[0]
        y = data[1]

        # Fit models and predict over training set
        lda_model = LDA()
        lda_model.fit(X, y)
        lda_predict = lda_model.predict(X)

        gaussian_model = GaussianNaiveBayes()
        gaussian_model.fit(X, y)
        gaussian_predict = gaussian_model.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        lda_acc = accuracy(y, lda_predict)
        gaussian_acc = accuracy(y, gaussian_predict)
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(f'Gaussian Naive Bayes, accuracy: {gaussian_acc}',
                                            f'LDA, accuracy: {lda_acc}'), x_title=0.5)

        # Add traces for data-points setting symbols and colors
        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
            marker=dict(color=gaussian_predict, symbol=y, size=4)), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
            marker=dict(color=lda_predict, symbol=y, size=4)), row=1, col=2)

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(go.Scatter(x=gaussian_model.mu_[:, 0], y=gaussian_model.mu_[:, 1], mode='markers',
                       marker=dict(color='black', symbol='x', size=10)), row=1, col=1)
        fig.add_trace(go.Scatter(x=lda_model.mu_[:, 0], y=lda_model.mu_[:, 1], mode='markers',
                       marker=dict(color='black', symbol='x', size=10)), row=1, col=2)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(len(gaussian_model.classes_)):
            fig.add_traces([get_ellipse(np.array(gaussian_model.mu_)[i, :],
                                        np.diag(np.array(gaussian_model.vars_)[i, :])), ], rows=1, cols=1)
        for i in range(len(lda_model.classes_)):
            fig.add_traces([get_ellipse(np.array(lda_model.mu_)[i, :], lda_model.cov_), ], rows=1, cols=2)

        fig.update_layout(title=f'Data Set: {f}', title_x=0.5, title_font_size=20,
                          width=1000, height=700, showlegend=False)
        fig.write_image(f'../ANSWERS/ex3/{f}.png')


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
