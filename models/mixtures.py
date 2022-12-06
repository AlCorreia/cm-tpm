from torch.distributions import Independent, Bernoulli, Normal, Categorical
import torch.nn as nn
import torch


class GaussianMixture(nn.Module):

    def __init__(
        self,
        mu,
        logvar,
        logits_w,
        min_std=0.01,
        max_std=1.5,
        learn_std: bool = True,
        learn_w: bool = True
    ):
        super(GaussianMixture, self).__init__()
        self.min_std = min_std
        self.max_std = max_std
        self.mu = nn.Parameter(mu, requires_grad=True)
        self.logvar = nn.Parameter(logvar, requires_grad=learn_std)
        self.logits_w = nn.Parameter(logits_w, requires_grad=learn_w)

    @property
    def std(self):
        return torch.clamp(torch.exp(0.5 * self.logvar), min=self.min_std, max=self.max_std)

    @property
    def log_w(self):
        return self.logits_w.log_softmax(0)

    def forward(self, x):
        dist = Independent(Normal(self.mu, self.std), x.ndim - 1)
        log_prob = dist.log_prob(x[:, None, ...])
        log_prob = torch.logsumexp(log_prob + self.log_w[None, ...], dim=1, keepdim=False)
        return log_prob

    def sample(self, n_samples: int):
        k = Categorical(logits=self.log_w).sample([n_samples])
        dist = Independent(Normal(self.mu[k], self.std[k]), self.mu.ndim - 1)
        samples = dist.sample()
        return samples


class BernoulliMixture(nn.Module):

    def __init__(
        self,
        logits_p,
        logits_w,
        learn_w: bool = True
    ):
        super(BernoulliMixture, self).__init__()
        self.logits_p = nn.Parameter(logits_p, requires_grad=True)
        self.logits_w = nn.Parameter(logits_w, requires_grad=learn_w)

    @property
    def p(self):
        return self.logits_p.sigmoid()

    @property
    def log_w(self):
        return self.logits_w.log_softmax(0)

    def forward(self, x):
        dist = Independent(Bernoulli(self.p), x.ndim - 1)
        log_prob = dist.log_prob(x[:, None, ...])
        log_prob = torch.logsumexp(log_prob + self.log_w[None, ...], dim=1, keepdim=False)
        return log_prob

    def sample(self, n_samples: int, return_p: bool = False):
        k = Categorical(logits=self.log_w).sample([n_samples])
        if return_p:
            samples = self.p[k]
        else:
            dist = Independent(Bernoulli(self.p[k]), self.p.ndim - 1)
            samples = dist.sample()
        return samples

