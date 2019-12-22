import torch
from torch import distributions, nn, Tensor

from bal.distributions import TruncatedNormal


class AdaptiveSize(nn.Module):
    def __init__(self,
                 min_size,
                 max_size,
                 prior_loc=1.0,
                 prior_scale=1.0,
                 temperature=1.0):
        super(AdaptiveSize, self).__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.temperature = torch.tensor(
            temperature,
            dtype=torch.float32,
            requires_grad=False
        )
        self._tril = torch.tril(
            torch.ones((self.max_size, self.max_size), requires_grad=False)
        )
        self._categories = torch.arange(
            self.min_size,
            self.max_size + 1,
            dtype=torch.float32
        )
        prior_scale_t = torch.tensor(prior_scale, dtype=torch.float32)
        prior_loc_t = torch.tensor(prior_loc, dtype=torch.float32)
        # compute the inverse softplus scale
        prior_spi_scale_t = torch.log(
            torch.exp(prior_scale_t) - 1.0
        )
        self.prior = self._build_prior(prior_loc_t, prior_spi_scale_t)
        self.var_loc = nn.Parameter(
            prior_loc_t,
            requires_grad=True
        )
        self.var_spi_scale = nn.Parameter(
            prior_spi_scale_t,
            requires_grad=True
        )
        self.posterior = self._build_posterior(self.var_loc, self.var_spi_scale)

    def _build_prior(self, init_loc, init_spi_scale):
        # build the prior distribution
        init_scale = nn.functional.softplus(init_spi_scale)
        tn_prior = TruncatedNormal(
            loc=init_loc,
            scale=init_scale,
            low=self.min_size,
            high=self.max_size
        )
        logits_prior = tn_prior.log_prob(self._categories)
        dist_prior = distributions.OneHotCategorical(logits=logits_prior)
        return dist_prior

    def _build_posterior(self, param_loc, param_spi_scale):
        tn_posterior = TruncatedNormal(
            loc=param_loc,
            scale=nn.functional.softplus(param_spi_scale),
            low=self.min_size,
            high=self.max_size
        )
        logits_posterior = tn_posterior.log_prob(self._categories)
        dist_posterior = distributions.RelaxedOneHotCategorical(
            logits=logits_posterior,
            temperature=self.temperature
        )
        return dist_posterior

    def forward(self, input: Tensor):
        self.posterior = self._build_posterior(
            param_loc=self.var_loc,
            param_spi_scale=self.var_spi_scale
        )
        # sample the size from the posterior distribution
        # and compute the soft mask from the lower triangular matrix
        post_sample = self.posterior.rsample(sample_shape=(input.shape[0],))
        mask = post_sample @ self._tril
        res = input * mask
        return res

    def extra_repr(self) -> str:
        return f"min_size={self.min_size}, max_size={self.max_size}"


class GaussianLikelihood(nn.Module):
    def __init__(self, mode="heteroscedastic"):
        super(GaussianLikelihood, self).__init__()
        self.mode = mode
        if mode == "homoscedastic":
            init_spi_scale = torch.log(
                torch.exp(torch.as_tensor(1.0, dtype=torch.float32)) - 1.0
            )
            self.var_spi_scale = nn.Parameter(
                init_spi_scale,
                requires_grad=True
            )
        elif mode == "heteroscedastic":
            self.var_spi_scale = None
        else:
            self.var_spi_scale = torch.log(
                torch.exp(torch.as_tensor(float(mode), dtype=torch.float32)) - 1.0
            )
        self.dist_lik = distributions.Normal

    def forward(self, *inputs):
        if self.mode == "heteroscedastic":
            spi_scale = inputs[1]
        else:
            spi_scale = self.var_spi_scale
        likelihood = self.dist_lik(
            inputs[0], nn.functional.softplus(spi_scale)
        )
        return likelihood
