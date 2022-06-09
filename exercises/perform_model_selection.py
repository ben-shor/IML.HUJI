from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X = np.random.uniform(-1.2, 2, n_samples)
    noiseless_y = np.array([(x+3)*(x+2)*(x+1)*(x-1)*(x-2) for x in X])
    noise_y = noiseless_y + np.random.normal(0, noise, n_samples)
    y = noise_y

    train_proportion = 2/3
    n_train = int(np.ceil(len(X) * train_proportion))
    shuffled_indexes = np.random.permutation(np.arange(len(X)))
    train_ids = shuffled_indexes[:n_train]
    test_ids = shuffled_indexes[n_train:]
    train_X, train_Y = X[train_ids], y[train_ids]
    test_X, test_Y = X[test_ids], y[test_ids]

    fig = go.Figure([go.Scatter(x=X, y=noiseless_y, mode="markers", name="noiseless", marker=dict(color="black")),
                     go.Scatter(x=train_X, y=train_Y, mode="markers", name="noisy train", marker=dict(color="red")),
                     go.Scatter(x=test_X, y=test_Y, mode="markers", name="noisy train", marker=dict(color="blue"))])
        # go.Scatter(x=X, y=y, mode="markers", name="noisy", marker=dict(color=colors))])
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_scores, validation_scores = [], []
    for k in range(11):
        poly_estimator = PolynomialFitting(k)
        train_err, validation_err = cross_validate(poly_estimator, train_X, train_Y, mean_square_error, cv=5)
        train_scores.append(train_err)
        validation_scores.append(validation_err)
    go.Figure([go.Scatter(x=list(range(len(train_scores))), y=train_scores, mode='markers+lines',
                          name=r'train error'),
               go.Scatter(x=list(range(len(validation_scores))), y=validation_scores, mode='markers+lines',
                          name=r'validation error')],
              layout=go.Layout(title="Loss for iteration number",
                               xaxis_title="Fitted polynomial degree",
                               yaxis_title="loss")).show()


    # # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = np.argmin(validation_scores)
    poly_estimator = PolynomialFitting(best_k)
    poly_estimator.fit(train_X, train_Y)
    test_predict = poly_estimator.predict(test_X)
    print(f"test error for degree {best_k}: {round(mean_square_error(test_predict, test_Y), 2)}")


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
    diabetes_dataset = datasets.load_diabetes()
    X = diabetes_dataset.data
    y = diabetes_dataset.target

    n_train = 50
    shuffled_indexes = np.random.permutation(np.arange(len(X)))
    train_ids = shuffled_indexes[:n_train]
    test_ids = shuffled_indexes[n_train:]
    train_X, train_Y = X[train_ids], y[train_ids]
    test_X, test_Y = X[test_ids], y[test_ids]


    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    train_scores, validation_scores = [], []
    lam_range = list(np.linspace(0, 3, 500))
    for lam in lam_range:
        regressor = RidgeRegression(lam)
        train_err, validation_err = cross_validate(regressor, train_X, train_Y, mean_square_error, cv=5)
        train_scores.append(train_err)
        validation_scores.append(validation_err)
    best_ridge = lam_range[np.argmin(validation_scores)]
    go.Figure([go.Scatter(x=lam_range, y=train_scores, mode='markers+lines',
                          name=r'train error'),
               go.Scatter(x=lam_range, y=validation_scores, mode='markers+lines',
                          name=r'validation error')],
              layout=go.Layout(title="Loss for Ridge regularization",
                               xaxis_title="Ridge lambda value",
                               yaxis_title="loss")).show()

    train_scores, validation_scores = [], []
    lam_range = list(np.linspace(0, 3, 500))
    for lam in lam_range:
        regressor = Lasso(lam)
        train_err, validation_err = cross_validate(regressor, train_X, train_Y, mean_square_error, cv=5)
        train_scores.append(train_err)
        validation_scores.append(validation_err)
    best_lasso = lam_range[np.argmin(validation_scores)]
    go.Figure([go.Scatter(x=lam_range, y=train_scores, mode='markers+lines',
                          name=r'train error'),
               go.Scatter(x=lam_range, y=validation_scores, mode='markers+lines',
                          name=r'validation error')],
              layout=go.Layout(title="Loss for Lasso regularization",
                               xaxis_title="Lasso lambda value",
                               yaxis_title="loss")).show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    regressor = RidgeRegression(best_ridge)
    regressor.fit(train_X, train_Y)
    test_predict = regressor.predict(test_X)
    print(f"Ridge - test error for lambda={best_ridge}: {round(mean_square_error(test_predict, test_Y), 2)}")

    regressor = Lasso(best_lasso)
    regressor.fit(train_X, train_Y)
    test_predict = regressor.predict(test_X)
    print(f"Lasso - test error for lambda={best_lasso}: {round(mean_square_error(test_predict, test_Y), 2)}")

    regressor = LinearRegression()
    regressor.fit(train_X, train_Y)
    test_predict = regressor.predict(test_X)
    print(f"LinearRegression - test error: {round(mean_square_error(test_predict, test_Y), 2)}")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()