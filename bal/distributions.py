import math
from pyro.distributions import Distribution
import torch

_SQRT2 = math.sqrt(2)


class TruncatedNormal(Distribution):
    def __init__(self, loc, scale, low, high, dtype=torch.float32):
        self._loc = torch.as_tensor(loc, dtype=dtype)
        self._scale = torch.as_tensor(scale, dtype=dtype)
        self._low = torch.as_tensor(low, dtype=dtype)
        self._high = torch.as_tensor(high, dtype=dtype)

    @property
    def loc(self):
        return self._loc

    @property
    def scale(self):
        return self._scale

    @property
    def low(self):
        return self._low

    @property
    def high(self):
        return self._high

    def sample(self, sample_shape=(1,)):
        # normalise the bounds of the uniform sampler
        low_uniform = _ndtr(self._z(self.low))
        high_uniform = _ndtr(self._z(self.high))
        aux = (high_uniform - low_uniform) * torch.rand(*sample_shape) + low_uniform
        samples = self._inv_z(_ndtri(aux))
        return samples

    def log_prob(self, x, **kwargs):
        broadcasted_x = x * torch.ones_like(self.loc)
        logits = torch.where(
            (broadcasted_x < self.low) | (broadcasted_x > self.high),
            torch.ones_like(broadcasted_x),
            self._log_unnormalised_prob(x) - self._log_normalisation()
        )
        return logits

    def _log_unnormalised_prob(self, x):
        broadcasted_x = x * torch.ones_like(self.loc)
        probs = torch.where(
            (broadcasted_x < self.low) | (broadcasted_x > self.high),
            torch.zeros_like(broadcasted_x),
            -0.5 * (self._z(broadcasted_x)**2)
        )
        return probs

    def _log_normalisation(self):
        """Compute the log normalisation constant of the PDF of
        the truncated normal distribution.
        """
        log_z = (0.5 * math.log(2. * math.pi)
                 + torch.log(self.scale)
                 + torch.log(self._interval_normalisation()))
        return log_z

    def _interval_normalisation(self):
        """Compute the constant due to the truncated support."""
        z_int = _ndtr(self._z(self.high)) - _ndtr(self._z(self.low))
        return z_int

    def _z(self, x):
        """Standardise the variable `x` in to zero mean and unit variance."""
        return (x - self.loc) / self.scale

    def _inv_z(self, z):
        """Invert the standardisation."""
        return z * self.scale + self.loc


def _ndtr(x):
    """CDF of standard normal evaluated at `x`."""
    return 0.5 * (1 + torch.erf(x / _SQRT2))


def _ndtri(x):
    """Inverse CDF of standard normal evaluated at `x`."""
    return torch.erfinv(2 * x - 1) * _SQRT2
