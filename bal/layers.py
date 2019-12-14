import numpy as np
import torch
from torch import distributions, nn, Tensor


class AdaptiveLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(AdaptiveLinear, self).__init__(in_features, out_features, bias)
        self.size_var = nn.Parameter(Tensor(), requires_grad=True)
        self.size_dist = distributions.RelaxedOneHotCategorical(
            temperature=self.temperature,
            logits=self.size_var
        )
        lower_triang = np.tril(np.ones(
            (self.out_features, self.out_features)))
        self._mask = torch.tensor(lower_triang)

    def forward(self, input: Tensor) -> Tensor:
        l_size = self._mask @ self.size_dist.sample()
        act = super(AdaptiveLinear, self).forward(input) * l_size
        return act

    def extra_repr(self) -> str:
        return super(AdaptiveLinear, self).extra_repr() + str(self.size_var)
