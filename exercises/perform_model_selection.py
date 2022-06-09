from __future__ import annotations
import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TITLE_X, TITLE_SIZE, WIDTH, HEIGHT = 0.5, 20, 1000, 700

def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """

    def f(x: np.ndarray):
        return (x+3)*(x+2)*(x+1)*(x-1)*(x-2)

    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X = np.linspace(-1.2, 2, n_samples)
    epsilons = np.random.normal(0, noise, n_samples)
    y = f(X)
    noisy_y = y + epsilons
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(noisy_y), 2/3)
    train_X, train_y, test_X, test_y = train_X.to_numpy(), train_y.to_numpy(), test_X.to_numpy(), test_y.to_numpy()
    fig = go.Figure(
        [go.Scatter(x=X, y=y, mode='lines+markers', marker=dict(color='black'), name='Noiseless Values'),
         go.Scatter(x=train_X[:, 0], y=train_y, mode='markers', name='Noisy Train values'),
         go.Scatter(x=test_X[:, 0], y=test_y, mode='markers', name='Noisy Test values')]
    )
    fig.update_layout(title_text=f'Model and Dataset Values With {n_samples} Samples and {noise} Noise',
                      title_x=TITLE_X, title_font_size=TITLE_SIZE, width=WIDTH, height=HEIGHT)
    fig.update_xaxes(title_text='x')
    fig.update_yaxes(title_text='f(x)')
    fig.write_image(f'../ANSWERS/ex5/question1_noise{noise}.png')

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    max_degree = 10
    train_error = []
    validation_error = []
    degrees = np.arange(max_degree + 1)
    for k in degrees:
        poly_model = PolynomialFitting(k)
        train_score, validation_score = cross_validate(poly_model, train_X, train_y, mean_square_error)
        train_error.append(train_score)
        validation_error.append(validation_score)
    fig = go.Figure(
        [go.Scatter(x=degrees, y=train_error, mode='lines+markers', name=r'Training Error'),
         go.Scatter(x=degrees, y=validation_error, mode='lines+markers', name=r'Validation Error')]
    )
    fig.update_layout(title_text=f'MSE of Training and Validation as a Function of the Polynom`s Degree, With {noise} '
                                 f'Noise', title_x=TITLE_X, title_font_size=TITLE_SIZE, width=WIDTH, height=HEIGHT)
    fig.update_xaxes(title_text='k-th degree')
    fig.update_yaxes(title_text='MSE')
    fig.write_image(f'../ANSWERS/ex5/question2_noise{noise}.png')

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = int(np.argmin(validation_error))
    best_model = PolynomialFitting(best_k)
    best_model.fit(train_X, train_y)
    test_error = round(best_model.loss(test_X, test_y), 2)
    print(f"\nThe degree that achieved the best polynomial model is: {best_k}, with error value of: {test_error}\n")

def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X, train_y = X[:n_samples, :], y[:n_samples]
    test_X, test_y = X[n_samples:, :], y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambdas = np.linspace(0.0001, 2.5, n_evaluations)
    ridge_training_errors = []
    ridge_validation_errors = []
    lasso_training_errors = []
    lasso_validation_errors = []
    for lam in lambdas:
        train_error, validation_error = cross_validate(RidgeRegression(lam), train_X, train_y, mean_square_error)
        ridge_training_errors.append(train_error)
        ridge_validation_errors.append(validation_error)
        train_error, validation_error = cross_validate(Lasso(lam), train_X, train_y, mean_square_error)
        lasso_training_errors.append(train_error)
        lasso_validation_errors.append(validation_error)

    fig = make_subplots(rows=2, cols=1, subplot_titles=['Ridge', 'Lasso'], x_title=0.5)
    fig.update_layout(title='Ridge and Lasso Errors as a Function of Lambda Parameter', title_x=TITLE_X,
                      title_font_size=TITLE_SIZE, width=WIDTH, height=HEIGHT)
    fig.add_traces([go.Scatter(x=lambdas, y=ridge_training_errors, name="Ridge Train"),
                    go.Scatter(x=lambdas, y=ridge_validation_errors, name="Ridge Validation")],
                   rows=1, cols=1)
    fig.add_traces([go.Scatter(x=lambdas, y=lasso_training_errors, name="Lasso Train"),
                    go.Scatter(x=lambdas, y=lasso_validation_errors, name="Lasso Validation")],
                   rows=2, cols=1)
    fig.write_image(f'../ANSWERS/ex5/question7.png')

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lam_ridge = round(lambdas[np.argmin(ridge_validation_errors)], 2)
    best_lam_lasso = round(lambdas[np.argmin(lasso_validation_errors)], 2)

    ridge_best = RidgeRegression(best_lam_ridge).fit(train_X, train_y)
    lasso_best = Lasso(best_lam_lasso).fit(train_X, train_y)
    ls = LinearRegression().fit(train_X, train_y)

    ridge_error = ridge_best.loss(test_X, test_y)
    lasso_error = mean_square_error(test_y, lasso_best.predict(test_X))
    ls_error = ls.loss(test_X, test_y)

    print(f"\nBest Ridge lambda parameter = {best_lam_ridge} with error = {ridge_error}\n"
          f"Best Lasso lambda parameter = {best_lam_lasso} with error = {lasso_error}\n"
          f"Best LS error = {ls_error}")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(1500, 10)
    select_regularization_parameter()
