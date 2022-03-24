import time
from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    real_mu = 10
    samples = np.random.normal(real_mu, 1, size=1000)
    gaussian_estimator = UnivariateGaussian(biased_var=False)
    gaussian_estimator.fit(samples)
    print((gaussian_estimator.mu_, gaussian_estimator.var_))


    # Question 2 - Empirically showing sample mean is consistent
    estimated_expectations_list = []
    ms = np.arange(10, 1001, 10)
    for m in ms:
        gaussian_estimator.fit(samples[:m])
        estimated_expectations_list.append(gaussian_estimator.mu_)

    estimated_expectations = np.array(estimated_expectations_list)
    estimated_expectations_distances = np.abs(estimated_expectations - real_mu)

    go.Figure([go.Scatter(x=ms, y=estimated_expectations_distances, mode='markers+lines',
                          name=r'$distance$')],
              layout=go.Layout(title=r"$\text{Distance from }\mu\text{ by number of samples used}$",
                               xaxis_title=r"$\text{number of samples}$",
                               yaxis_title="r$Distance from \mu$",
                               height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    sorted_samples = np.sort(samples)
    pdfs = gaussian_estimator.pdf(sorted_samples)
    go.Figure([go.Scatter(x=sorted_samples, y=pdfs, mode='markers+lines',
                          name=r'$pdf for x value$')],
              layout=go.Layout(title=r"$\text{probability density function by sample value}$",
                               xaxis_title=r"$\text{samples value}$",
                               yaxis_title=r"$\text{probability density function value}$",
                               height=300)).show()

def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    real_mu = [0, 0, 4, 0]
    real_sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(real_mu, real_sigma, size=1000)
    gaussian_estimator = MultivariateGaussian()
    gaussian_estimator.fit(samples)
    print(gaussian_estimator.mu_)
    print(gaussian_estimator.cov_)

    # Question 5 - Likelihood evaluation
    heatmap = []
    start = time.time()
    linspace = np.linspace(-10, 10, 200)
    for f3 in linspace:
        heatmap.append([])
        for f1 in linspace:
            heatmap[-1].append(gaussian_estimator.log_likelihood(mu=np.array([f1, 0, f3, 0]), cov=real_sigma, X=samples))
        # print(f"took until now: {time.time()-start} {f3}")
    heatmap = np.array(heatmap)

    fig = px.imshow(heatmap, x=linspace, y=linspace, labels=dict(x="f1", y="f3", color="log-likelihood"))
    fig.show()


    # Question 6 - Maximum likelihood
    best_inds = np.where(heatmap == np.amax(heatmap))
    print(f"best log likelihood={np.amax(heatmap)} for "
          f"f1={round(linspace[best_inds[1][0]], 3)} f3={round(linspace[best_inds[0][0]], 3)}")


if __name__ == '__main__':
    np.random.seed(0)
    # samples = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
    #                     -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
    # gaussian_estimator = UnivariateGaussian(biased_var=False)
    # print(gaussian_estimator.log_likelihood(10,1,samples))

    test_univariate_gaussian()
    test_multivariate_gaussian()
