import torch
from torch import distributions, nn, Tensor

from bal.distributions import TruncatedNormal


class AdaptiveSize(nn.Module):
    def __init__(self,
                 min_size,
                 max_size,
                 prior_size_loc=1.0,
                 prior_size_scale=1.0,
                 temperature=1.0):
        super(AdaptiveSize, self).__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.temperature = temperature
        prior_size_scale_t = torch.tensor(prior_size_scale, dtype=torch.float32)
        # compute the inverse softplus scale
        prior_spi_scale = torch.log(torch.exp(prior_size_scale_t) - 1.0)
        self.posterior, self.prior = self._build_qp(
            init_loc=prior_size_loc,
            init_spi_scale=prior_spi_scale
        )
        self._tril = torch.tril(
            torch.ones((self.max_size, self.max_size))
        )

    def _build_qp(self, init_loc, init_spi_scale):
        # build the prior distribution
        init_scale = nn.functional.softplus(init_spi_scale)
        tn_prior = TruncatedNormal(
            loc=init_loc,
            scale=init_scale,
            low=self.min_size,
            high=self.max_size
        )
        size_cats = torch.arange(self.min_size, self.max_size + 1, dtype=torch.float32)
        logits_prior = tn_prior.log_prob(size_cats)
        dist_prior = distributions.OneHotCategorical(logits=logits_prior)

        var_loc = torch.tensor(init_loc, dtype=torch.float32)
        self.var_loc = nn.Parameter(var_loc, requires_grad=True)
        var_spi_scale = init_spi_scale
        self.var_spi_scale = nn.Parameter(var_spi_scale, requires_grad=True)
        tn_posterior = TruncatedNormal(
            loc=self.var_loc,
            scale=nn.functional.softplus(self.var_spi_scale),
            low=self.min_size,
            high=self.max_size
        )
        logits_posterior = tn_posterior.log_prob(size_cats)
        dist_posterior = distributions.RelaxedOneHotCategorical(
            logits=logits_posterior,
            temperature=self.temperature
        )
        return dist_posterior, dist_prior

    def forward(self, input: Tensor) -> Tensor:
        # sample the size from the posterior distribution
        # and compute the soft mask from the lower triangular matrix
        mask = self._tril @ self.posterior.sample()
        res = input * mask
        return res

    def extra_repr(self) -> str:
        return f"min_size={self.min_size}, max_size={self.max_size}"
