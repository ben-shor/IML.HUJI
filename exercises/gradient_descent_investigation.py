import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type
from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import mean_square_error
from IMLearn.model_selection import cross_validate
from IMLearn.utils import split_train_test

import plotly.graph_objects as go
from sklearn.metrics import auc


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights_history = []
    def _callback(weights, val, **kwargs):
        values.append(val)
        weights_history.append(weights)
    return _callback, values, weights_history


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    _callback, values, weights_history = get_gd_state_recorder_callback()
    gd = GradientDescent(learning_rate=FixedLR(1e-4),
                         max_iter=20000,
                         callback=_callback)

    mod = L1(init)
    gd.fit(mod, None, None)
    plot_descent_path(module=L1, descent_path=np.array(weights_history), title="L1 descent").show()

    _callback, values, weights_history = get_gd_state_recorder_callback()
    gd = GradientDescent(learning_rate=FixedLR(1e-4),
                         max_iter=20000,
                         callback=_callback)

    mod = L2(init)
    gd.fit(mod, None, None)
    plot_descent_path(module=L2, descent_path=np.array(weights_history), title="L2 descent").show()

    fig = go.Figure()
    for eta in etas:
        _callback, values, weights_history = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=FixedLR(eta),
                             max_iter=2000,
                             callback=_callback)

        gd.fit(L1(init), None, None)
        fig.add_scatter(x=list(range(len(values))), y=values, mode='markers+lines', name=f'eta={eta}')
    fig.update_layout(title=f"L1")
    fig.show()

    fig = go.Figure()
    for eta in etas:
        _callback, values, weights_history = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=FixedLR(eta),
                             max_iter=2000,
                             callback=_callback)

        gd.fit(L2(init), None, None)
        fig.add_scatter(x=list(range(len(values))), y=values, mode='markers+lines', name=f'eta={eta}')
    fig.update_layout(title=f"L2")
    fig.show()


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    # Plot algorithm's convergence for the different values of gamma
    fig = go.Figure()
    for gamma in gammas:
        _callback, values, weights_history = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=ExponentialLR(eta, gamma),
                             max_iter=2000,
                             callback=_callback)

        gd.fit(L1(init), None, None)
        fig.add_scatter(x=list(range(len(values))), y=values, mode='markers+lines', name=f'gamma={gamma}')
    fig.update_layout(title=f"L1")
    fig.show()

    # Plot descent path for gamma=0.95
    _callback, values, weights_history = get_gd_state_recorder_callback()
    gd = GradientDescent(learning_rate=ExponentialLR(0.1, 0.95),
                         max_iter=2000,
                         callback=_callback)
    mod = L1(init)
    gd.fit(mod, None, None)
    plot_descent_path(module=L1, descent_path=np.array(weights_history), title="L1 descent").show()

def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # Plotting convergence rate of logistic regression over SA heart disease data
    lr = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000))
    lr.fit(X_train, y_train)
    y_pred_prob = lr.predict_proba(X_test)

    fpr, tpr = [], []
    alphas = []
    for i in range(101):
        alpha = i / 100
        tp = np.sum((y_pred_prob > alpha) & y_test)
        fp = np.sum((y_pred_prob > alpha) & ~y_test)
        tn = np.sum((y_pred_prob <= alpha) & ~y_test)
        fn = np.sum((y_pred_prob <= alpha) & y_test)

        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))
        alphas.append((alpha, tpr[-1] - fpr[-1]))
    alphas_str = list(map(str, alphas))
    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'), name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=alphas_str, name="", showlegend=False, marker_size=5,
                         hovertemplate="<b>Threshold:</b>%{text}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()

    best_alpha = max(alphas, key=lambda x: x[1])
    print("best alpha: ", best_alpha)
    print(f"test error for alpha={best_alpha[0]}: ", {round(mean_square_error((y_pred_prob > best_alpha[0]), y_test), 3)})

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    best_lam = best_lam_score = None
    for lam in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:
        lr = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000),
                                penalty="l1", lam=lam)
        train_err, validation_err = cross_validate(lr, X_train, y_train, mean_square_error, cv=5)
        if best_lam_score is None or validation_err < best_lam_score:
            best_lam = lam
            best_lam_score = validation_err
        print(f"score for L1 {lam} is {validation_err}")
    print(f"best lam L1: {best_lam}")

    lr = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000),
                            penalty="l1", lam=best_lam)
    lr.fit(X_train, y_train)
    y_pred = lr.predict_proba(X_test)
    print(f"test error L1 with lam={best_lam}: ", {round(mean_square_error(y_pred, y_test), 3)})

    # Same but with L2
    best_lam = best_lam_score = None
    for lam in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:
        lr = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000),
                                penalty="l2", lam=lam)
        train_err, validation_err = cross_validate(lr, X_train, y_train, mean_square_error, cv=5)
        if best_lam_score is None or validation_err < best_lam_score:
            best_lam = lam
            best_lam_score = validation_err
        print(f"score for L2 {lam} is {validation_err}")
    print(f"best lam L2: {best_lam}")

    lr = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000),
                            penalty="l1", lam=best_lam)
    lr.fit(X_train, y_train)
    y_pred = lr.predict_proba(X_test)
    print(f"test error L2 with lam={best_lam}: ", {round(mean_square_error(y_pred, y_test), 3)})


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
