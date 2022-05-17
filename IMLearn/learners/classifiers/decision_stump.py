from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from ...metrics import misclassification_error


def misclassification_error_weighted(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    y_pred is -1/1 and y_true is a weighted value, with the sign representing the classification and the magnitude
    representing the weight of error
    """
    not_correct = np.sign(y_true) != np.sign(y_pred)
    return np.sum(np.abs(y_true) * not_correct)


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        best_feature = best_thr = best_err = best_sign = None
        for feature in range(X.shape[1]):
            err1, thr1 = self._find_threshold(X[:, feature], y, -1)
            err2, thr2 = self._find_threshold(X[:, feature], y, 1)
            # print(thr1, err1, thr2, err2, err1+err2)
            if best_err is None or err1 < best_err:
                best_feature, best_thr, best_err, best_sign = feature, thr1, err1, -1
            if err2 < best_err:
                best_feature, best_thr, best_err, best_sign = feature, thr2, err2, 1
        self.j_ = best_feature
        self.sign_ = best_sign
        self.threshold_ = best_thr


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        prediction = (X[:, self.j_] > self.threshold_).astype(int)
        prediction[prediction == 0] = -1
        return self.sign_ * prediction

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        v = values.copy()
        sorted_indexes = np.argsort(v)

        # start the error as if the threshold is -inf and everything is classified as sign
        best_err = err = misclassification_error_weighted(y_true=labels, y_pred=sign * np.ones(len(v)))
        best_thr = min(0, v[sorted_indexes[0]] * 2, v[sorted_indexes[0]] / 2)
        for i in range(len(sorted_indexes)):
            # we change the classification of the i'th sorted index,
            if sign == np.sign(labels[sorted_indexes[i]]):
                err += np.abs(labels[sorted_indexes[i]])
            else:
                err -= np.abs(labels[sorted_indexes[i]])
            if err < best_err:
                if i != len(sorted_indexes) - 1:
                    best_thr = (v[sorted_indexes[i]] + v[sorted_indexes[i + 1]]) / 2
                else:
                    best_thr = max(0, v[sorted_indexes[i]] * 2)
                best_err = err
        return best_err, best_thr

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
        return misclassification_error(y, self.predict(X))
