from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


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
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        percepton = Perceptron(callback=lambda p, a, b: losses.append(p.loss(X, y)))
        percepton.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        go.Figure([go.Scatter(x=list(range(len(losses))), y=losses, mode='markers+lines',
                              name=r'$loss$')],
                  layout=go.Layout(title="loss for iteration number",
                                   xaxis_title="iteration number",
                                   yaxis_title="loss")).show()


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

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black", showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit models and predict over training set
        lda_classifier = LDA()
        lda_classifier.fit(X, y)
        lda_predict = lda_classifier.predict(X)

        gnb_classifier = GaussianNaiveBayes()
        gnb_classifier.fit(X, y)
        gnb_predict = gnb_classifier.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.01, vertical_spacing=.03,
                            subplot_titles=[f"LDA classifier, accuracy: {accuracy(y, lda_predict)}",
                                            f"Gaussian Naive Bayes classifier, accuracy: {accuracy(y, gnb_predict)}"])

        fig.update_layout(title=f"Classifiers for dataset: {f}")


        # Add traces for data-points setting symbols and colors
        fig.add_trace(
            go.Scatter(x=X[:,0], y=X[:,1], mode="markers", showlegend=False,
                       marker=dict(color=lda_predict, symbol=y, line=dict(color="black", width=1))),
            1, 1)

        fig.add_trace(
            go.Scatter(x=X[:,0], y=X[:,1], mode="markers", showlegend=False,
                       marker=dict(color=gnb_predict, symbol=y, line=dict(color="black", width=1))),
            1, 2)

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(
            go.Scatter(x=lda_classifier.mu_[:,0], y=lda_classifier.mu_[:,1], mode="markers", showlegend=False,
                       marker=dict(color="black", symbol="x", size=20)),
            1, 1)

        fig.add_trace(
            go.Scatter(x=gnb_classifier.mu_[:,0], y=gnb_classifier.mu_[:,1], mode="markers", showlegend=False,
                       marker=dict(color="black", symbol="x", size=20)),
            1, 2)

        # Add ellipses depicting the covariances of the fitted Gaussians
        fig.add_trace(get_ellipse(lda_classifier.mu_[0], lda_classifier.cov_), 1, 1)
        fig.add_trace(get_ellipse(lda_classifier.mu_[1], lda_classifier.cov_), 1, 1)
        fig.add_trace(get_ellipse(lda_classifier.mu_[2], lda_classifier.cov_), 1, 1)

        fig.add_trace(get_ellipse(gnb_classifier.mu_[0], np.diag(gnb_classifier.vars_[0])), 1, 2)
        fig.add_trace(get_ellipse(gnb_classifier.mu_[1], np.diag(gnb_classifier.vars_[1])), 1, 2)
        fig.add_trace(get_ellipse(gnb_classifier.mu_[2], np.diag(gnb_classifier.vars_[2])), 1, 2)

        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
