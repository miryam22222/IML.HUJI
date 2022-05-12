import numpy as np
from typing import Tuple

import utils
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics.loss_functions import accuracy
TITLE_X, TITLE_SIZE, WIDTH, HEIGHT = 0.5, 20, 1000, 700

def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def scatter_test_samples(X, y, point_size):
    return go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                      marker=dict(color=y, colorscale=[custom[0], custom[-1]], line=dict(color="black", width=1),
                                  size=point_size))


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ada_boost = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    train_results = []
    test_results = []
    n_learners_list = np.arange(n_learners)
    for i in n_learners_list:
        train_results.append(ada_boost.partial_loss(train_X, train_y, i))
        test_results.append(ada_boost.partial_loss(test_X, test_y, i))
    fig = go.Figure([go.Scatter(x=n_learners_list, y=train_results, mode='lines', name='Train Loss'),
                     go.Scatter(x=n_learners_list, y=test_results, mode='lines', name='Test Loss')])
    fig.update_xaxes(title_text='Iterations')
    fig.update_yaxes(title_text='Errors')
    fig.update_layout(title_text=f'Training and Test Errors as a Function of Fitted Learners With {noise} Noise',
                      title_x=TITLE_X, title_font_size=TITLE_SIZE, width=WIDTH, height=HEIGHT)
    fig.write_image(f'../ANSWERS/ex4/question1_{noise}_noise.png')

    # # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f'{t} Iterations' for t in T], x_title=0.5)
    fig.update_layout(title=f'Decision Boundaries With {noise} Noise', title_x=TITLE_X, title_font_size=TITLE_SIZE,
                      width=WIDTH, height=HEIGHT, showlegend=False)
    fig.update_xaxes(range=[-1, 1])
    fig.update_yaxes(range=[-1, 1])
    for i, t in enumerate(T):
        decision_surface_trace = decision_surface(lambda x: ada_boost.partial_predict(X=x, T=t), lims[0], lims[1],
                                                  showscale=False)
        test_sample = scatter_test_samples(test_X, test_y, np.ones(train_size) * 5)
        fig.add_traces([decision_surface_trace, test_sample], rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig.write_image(f'../ANSWERS/ex4//question2_{noise}_noise.png')

    # Question 3: Decision surface of best performing ensemble
    mse_index = int(np.argmin(test_results))
    best_decision_surface = decision_surface(lambda x: ada_boost.partial_predict(X=x, T=mse_index), lims[0], lims[1],
                                             showscale=False)
    test_sample = scatter_test_samples(test_X, test_y, np.ones(train_size)*5)
    fig = go.Figure([best_decision_surface, test_sample])
    fig.update_layout(title=f'Decision Boundaries of {mse_index} Iterations, {noise} Noise and '
                            f'{accuracy(test_y, ada_boost.partial_predict(test_X, mse_index))} Accuracy',
                      title_x=TITLE_X, title_font_size=TITLE_SIZE, width=WIDTH, height=HEIGHT, showlegend=False)
    fig.update_xaxes(range=[-1, 1])
    fig.update_yaxes(range=[-1, 1])
    fig.write_image(f'../ANSWERS/ex4/question3_{noise}_noise.png')

    # Question 4: Decision surface with weighted samples
    point_sizes = ada_boost.D_ / np.max(ada_boost.D_) * 5
    full_decision_surface = decision_surface(lambda x: ada_boost.predict(X=x), lims[0], lims[1],
                                             showscale=False)
    train_sample = scatter_test_samples(train_X, train_y, point_sizes)
    fig = go.Figure([full_decision_surface, train_sample])
    fig.update_layout(title=f'Decision Boundaries of {n_learners} Iterations, {noise} Noise and '
                            f'{accuracy(test_y, ada_boost.partial_predict(test_X, n_learners))} Accuracy',
                      title_x=TITLE_X, title_font_size=TITLE_SIZE, width=WIDTH, height=HEIGHT, showlegend=False)
    fig.update_xaxes(range=[-1, 1])
    fig.update_yaxes(range=[-1, 1])
    fig.write_image(f'../ANSWERS/ex4/question4_{noise}_noise.png')


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)