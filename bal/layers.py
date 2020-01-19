import torch
from torch import distributions, nn, Tensor

from bal.distributions import TruncatedNormal


_LOG_EPSILON = -18.4207     # == log(1e-8)


class BayesianLayer(nn.Module):
    def __init__(self):
        super(BayesianLayer, self).__init__()

    def _build_prior(self, *args):
        pass

    def _build_posterior(self, *args):
        pass

    def kl_posterior_prior(self, n_samples=1):
        samples_q = self.posterior.rsample((n_samples,))
        kl = torch.mean(
            self.posterior.log_prob(samples_q) - self.prior.log_prob(samples_q)
        )
        return kl


class AdaptiveSize(BayesianLayer):
    def __init__(self,
                 min_size,
                 max_size,
                 prior_loc=1.0,
                 prior_scale=1.0,
                 temperature=1.0,
                 prior_temperature=1e-3):
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
        self.prior = self._build_prior(prior_loc_t, prior_spi_scale_t, prior_temperature)
        self.var_loc = nn.Parameter(
            prior_loc_t,
            requires_grad=True
        )
        self.var_spi_scale = nn.Parameter(
            prior_spi_scale_t,
            requires_grad=True
        )
        self.posterior = self._build_posterior(self.var_loc, self.var_spi_scale)

    def _build_prior(self, prior_loc, prior_spi_scale, prior_temperature):
        tn_prior = TruncatedNormal(
            loc=prior_loc,
            scale=torch.clamp_min(nn.functional.softplus(prior_spi_scale), 1e-3),
            low=self.min_size,
            high=self.max_size
        )
        logits_prior = tn_prior.log_prob(self._categories)
        dist_prior = distributions.RelaxedOneHotCategorical(
            logits=torch.clamp_min(logits_prior, _LOG_EPSILON),
            temperature=torch.tensor(prior_temperature, dtype=torch.float32)
        )
        return dist_prior

    def _build_posterior(self, param_loc, param_spi_scale):
        tn_posterior = TruncatedNormal(
            loc=torch.clamp(param_loc, self.min_size - 1, self.max_size + 1),
            scale=torch.clamp_min(nn.functional.softplus(param_spi_scale), 1e-3),
            low=self.min_size,
            high=self.max_size
        )
        logits_posterior = tn_posterior.log_prob(self._categories)
        dist_posterior = distributions.RelaxedOneHotCategorical(
            logits=torch.clamp_min(logits_posterior, _LOG_EPSILON),
            temperature=self.temperature
        )
        return dist_posterior

    def forward(self, input: Tensor):
        self.posterior = self._build_posterior(
            param_loc=self.var_loc,
            param_spi_scale=self.var_spi_scale
        )
        # sample the size from the posterior distribution
        # and compute a soft mask from a lower triangular matrix
        post_sample = self.posterior.rsample()
        mask = post_sample @ self._tril
        res = input * mask
        return res

    def extra_repr(self) -> str:
        return f"min_size={self.min_size}, max_size={self.max_size}"


class SkipConnection(BayesianLayer):
    def __init__(self,
                 prior_prob=0.5,
                 temperature=1.0,
                 prior_temperature=1e-3):
        super(SkipConnection, self).__init__()
        self.temperature = torch.tensor(
            temperature,
            dtype=torch.float32,
            requires_grad=False
        )
        prior_prob_t = torch.tensor(prior_prob, dtype=torch.float32)
        prior_logit_t = torch.log(prior_prob_t) - torch.log(1.0 - prior_prob_t)
        self.prior = self._build_prior(prior_logit_t, prior_temperature)
        self.var_logit = nn.Parameter(
            prior_logit_t,
            requires_grad=True
        )
        self.posterior = self._build_posterior(self.var_logit)

    def _build_prior(self, prior_logit, prior_temperature):
        dist_prior = distributions.RelaxedBernoulli(
            logits=torch.clamp_min(prior_logit, _LOG_EPSILON),
            temperature=torch.tensor(prior_temperature, dtype=torch.float32)
        )
        return dist_prior

    def _build_posterior(self, param_logit):
        dist_posterior = distributions.RelaxedBernoulli(
            logits=param_logit,
            temperature=self.temperature
        )
        return dist_posterior

    def forward(self, input: Tensor, output: Tensor):
        self.posterior = self._build_posterior(self.var_logit)
        skip_prob = self.posterior.rsample()
        res = skip_prob * input + (1.0 - skip_prob) * output
        return res


class GaussianLikelihood(nn.Module):
    def __init__(self, scale="heteroscedastic"):
        super(GaussianLikelihood, self).__init__()
        self.scale = scale
        if scale == "homoscedastic":
            init_spi_scale = torch.log(
                torch.exp(torch.as_tensor(1.0, dtype=torch.float32)) - 1.0
            )
            self.var_spi_scale = nn.Parameter(
                init_spi_scale,
                requires_grad=True
            )
        elif scale == "heteroscedastic":
            self.var_spi_scale = None
        else:
            self.var_spi_scale = torch.log(
                torch.exp(torch.as_tensor(float(scale), dtype=torch.float32)) - 1.0
            )
        self.dist_lik = distributions.Normal

    def forward(self, *inputs):
        if self.scale == "heteroscedastic":
            spi_scale = inputs[1]
        else:
            spi_scale = self.var_spi_scale
        likelihood = self.dist_lik(
            inputs[0], nn.functional.softplus(spi_scale)
        )
        return likelihood

    def extra_repr(self) -> str:
        return f"scale type={self.scale}"


class CategoricalLikelihood(nn.Module):
    def __init__(self):
        super(CategoricalLikelihood, self).__init__()
        self.dist_lik = distributions.OneHotCategorical

    def forward(self, inputs):
        likelihood = self.dist_lik(
            logits=inputs
        )
        return likelihood
