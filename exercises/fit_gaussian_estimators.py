from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.io as pio
from matplotlib import pyplot as pt
from matplotlib.pyplot import figure

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    ACTUAL_MU = 10
    X = np.random.normal(10, 1, 1000)
    gausian_samples_1 = UnivariateGaussian()
    gausian_samples_1.fit(X)
    print(gausian_samples_1)

    # Question 2 - Empirically showing sample mean is consistent
    SAMPLES = 1000
    estimated_mean = np.empty(0)
    gausian_samples_2 = UnivariateGaussian()
    for m in range(10, SAMPLES + 1, 10):
        gausian_samples_2.fit(X[:m + 1])
        estimated_mean = np.append(estimated_mean, gausian_samples_2.mu_)
    estimated_mean_absolutes = np.abs(estimated_mean - ACTUAL_MU)

    figure(figsize=(12, 6), dpi=80)
    pt.plot(range(10, SAMPLES + 1, 10), estimated_mean_absolutes)
    pt.xlabel("Number of Samples", size=15)
    pt.ylabel("Differance Between True and Estimated Expectation", size=12)
    pt.title("Estimation of Expectation As Function of Number of Samples",
             size=20)
    pt.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf_samples = gausian_samples_1.pdf(X)
    pt.scatter(X, pdf_samples, s=1)
    pt.xlabel("The Value of the Samples")
    pt.ylabel("PDF of the Samples")
    pt.title("Normal Distribution of the samples")
    pt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.array(
        [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mu, cov, 1000)
    multivariate_gaussian_samples = MultivariateGaussian()
    multivariate_gaussian_samples.fit(X)
    print(multivariate_gaussian_samples)

    # Question 5 - Likelihood evaluation
    z = []
    f1_range = np.linspace(-10, 10, 200)
    f3_range = np.linspace(-10, 10, 200)
    max_likelihood = float("-inf")
    max_f1 = 0
    max_f3 = 0
    for f1 in f1_range:
        row = []
        for f3 in f3_range:
            mu_ = np.array([f1, 0, f3, 0])
            value = MultivariateGaussian.log_likelihood(mu_, cov, X)
            if value > max_likelihood:
                max_likelihood = value
                max_f1 = f1
                max_f3 = f3
            row.append(value)
        z.append(row)
    x, y = np.meshgrid(f1_range, f3_range)
    fig, ax = pt.subplots()
    c = ax.pcolormesh(x, y, z, cmap='RdBu')
    ax.set(title="Log Likelihood for mu = [f1, 0, f3, 0]",
           xlabel="f3",
           ylabel="f1")
    fig.colorbar(c, ax=ax)
    pt.show()

    # Question 6 - Maximum likelihood
    print(max_f1, max_f3)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
