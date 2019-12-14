import numpy as np


def generate_1d_regression(
        n_points=100,
        domain=(-5.0, 5.0),
        noise_std=0.0,
        seed=None):

    def f(x):
        x = np.atleast_1d(x)
        y = 0.3 * x + (0.15 * x + 0.3) * np.sin(np.pi * x)
        return y

    rng = np.random.RandomState(seed)
    xs = rng.uniform(*domain, size=n_points)
    ys = f(xs) + rng.normal(0.0, noise_std, size=n_points)
    return xs, ys
