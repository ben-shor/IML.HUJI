from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    folds_X = []
    folds_y = []
    fold_size = int(np.ceil(X.shape[0] / cv))
    for i in range(cv):
        folds_X.append(X[i*fold_size:(i+1)*fold_size])
        folds_y.append(y[i*fold_size:(i+1)*fold_size])

    train_scores = []
    test_scores = []
    for i in range(cv):
        train_X = np.concatenate(folds_X[:i] + folds_X[i+1:])
        train_y = np.concatenate(folds_y[:i] + folds_y[i+1:])
        test_X = folds_X[i]
        test_y = folds_y[i]

        # print(len(train_X), len(test_X))

        estimator.fit(train_X, train_y)
        train_predict = estimator.predict(train_X)
        test_predict = estimator.predict(test_X)

        train_scores.append(scoring(train_y, train_predict))
        test_scores.append(scoring(test_y, test_predict))
    return (sum(train_scores) / cv, sum(test_scores) / cv)
