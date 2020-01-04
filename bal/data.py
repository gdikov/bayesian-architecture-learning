import numpy as np


def generate_1d_regression(
        n_points=2000,
        domain=(-2.0, 2.0),
        noise_std=0.1,
        seed=None):

    def f(x):
        x = np.atleast_1d(x)
        y = np.sin(6 * x) + 0.4 * x**2 - 0.1 * x**3 - x * np.cos(9 * np.sqrt(np.exp(x)))
        # y = 0.3 * x + (0.15 * x + 0.3) * np.sin(np.pi * x)
        return y

    rng = np.random.RandomState(seed)
    xs = rng.uniform(*domain, size=n_points).astype(np.float32).reshape(-1, 1)
    ys = (f(xs) + rng.normal(0.0, noise_std, size=(n_points, 1))).astype(np.float32)
    return xs, ys


def batch_generator(xs, ys, batch_size, shuffle=False, seed=None):
    """Batch generator of a labeled dataset.

    Notes:
        If len(xs) % batch_size > 0 the batches will not be of equal length
        and may even be larger than batch_size.
    """
    if len(xs) != len(ys):
        raise ValueError("Length of xs and ys is not the same.")

    rng = np.random.RandomState(seed)
    indices = np.arange(len(xs))
    if shuffle:
        rng.shuffle(indices)

    n_batches = len(xs) // batch_size
    batch_indices = np.array_split(indices, n_batches)
    for ids in batch_indices:
        yield xs[ids], ys[ids]
