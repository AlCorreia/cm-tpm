from typing import Optional
from numbers import Real
import torch
import math
from torch.distributions import Categorical


def bce_loss(
    logits_p: torch.Tensor,
    x: torch.Tensor,
    missing: Optional[bool] = None,
    dim_start_sum: Optional[int] = 2,
    eps: float = 1e-7
):
    x = x.unsqueeze(1)
    if missing is None or missing is True:
        mask = ~ torch.isnan(x)
    else:
        mask = torch.ones_like(x)

    x = torch.nan_to_num(x)
    p = torch.clip(logits_p.sigmoid(), eps, 1 - eps)
    log_prob = x * torch.log(p + eps) + (1 - x) * torch.log(1 - p + eps)
    log_prob = log_prob * mask

    if dim_start_sum is not None:
        log_prob = log_prob.sum(dim=[i for i in range(dim_start_sum, log_prob.ndim)])
    return log_prob


def mse_loss(
    loc: torch.Tensor,
    scale: torch.Tensor,
    x: torch.Tensor,
    missing: Optional[bool] = None,
    dim_start_sum: Optional[int] = 2,
):
    x = torch.nan_to_num(x)
    x = x.unsqueeze(1)
    var = (scale ** 2)
    log_scale = math.log(scale) if isinstance(scale, Real) else scale.log()
    log_prob = -((x - loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
    if missing is None or missing is True:
        mask = ~ torch.isnan(x)
        log_prob = log_prob * mask
    if dim_start_sum is not None:
        log_prob = log_prob.sum(dim=[i for i in range(2, log_prob.ndim)])
    return log_prob


def ce_loss(
    logits_p: torch.Tensor,
    x: torch.Tensor,
    k: Optional[int] = None,
    missing: Optional[bool] = None,
    dim_start_sum: Optional[int] = 2,
):

    batch_size, data_dim = x.shape[0], x.shape[1:]
    n_bins = logits_p.shape[0]
    logits_p = logits_p.view(n_bins, -1, *data_dim)

    if missing is None or missing is True:
        mask = ~ torch.isnan(x)
    else:
        mask = torch.ones_like(x)
    x = torch.nan_to_num(x)

    if k is None:
        k = n_bins  # k is actually equal to n_bins in this case
        # Categorical probs as last dimension
        logits_p = torch.movedim(logits_p, 1, -1)
        # Create categorical distribution
        dist = Categorical(logits=logits_p)
        # Compute log-probs for all combinations of x and z (logits) values
        # This is done by unsqueezing the first dimension in x
        log_prob = dist.log_prob(x.unsqueeze(1).float())
        # Expand mask to broadcast to every pair (x, z)
        mask = mask.unsqueeze(1)
        # The first two dimensions are batch_size and n_bins.
        # The remaining dimensions are independent, and so we sum their log_prob
        # starting at dimension 2
        dim_start_sum = 2
    else:
        assert batch_size == int(n_bins / k)
        # Categorical probs as last dimension
        logits_p = torch.movedim(logits_p, 1, -1)
        # Create categorical distribution
        dist = Categorical(logits=logits_p)
        # Compute log-probs for each of x against corresponding k logits
        # This is done by expanding the first dimension of x to batch_size * k
        mask = mask.repeat_interleave(k, dim=0)
        x = x.repeat_interleave(k, dim=0)
        log_prob = dist.log_prob(x.float())
        # The first dimension is batch_size * k.
        # The remaining dimensions are independent, and so we sum their log_prob
        # starting at dimension 1
        dim_start_sum = 1

    log_prob = log_prob * mask
    if dim_start_sum is not None:
        log_prob = log_prob.sum(dim=[i for i in range(dim_start_sum, log_prob.ndim)])
    return log_prob.view(batch_size, k)

