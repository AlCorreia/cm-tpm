from typing import Optional
from numbers import Real
import torch
import math
from torch.distributions import Independent, Categorical


def bce(
    logits_p: torch.Tensor,
    x: torch.Tensor,
    missing: Optional[bool] = None,
    dim_start_sum: Optional[int] = 2,
    eps: float = 1e-7
):
    x = x.unsqueeze(1)
    x = torch.nan_to_num(x)
    p = torch.clip(logits_p.sigmoid(), eps, 1 - eps)
    log_prob = x * torch.log(p + eps) + (1 - x) * torch.log(1 - p + eps)

    if missing is None or missing is True:
        mask = ~ torch.isnan(x)
        log_prob = log_prob * mask

    if dim_start_sum is not None:
        log_prob = log_prob.sum(dim=[i for i in range(dim_start_sum, log_prob.ndim)])
    return log_prob


def mse(
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


def ce(
    logits_p: torch.Tensor,
    x: torch.Tensor,
    missing: Optional[bool] = None,
    dim_start_sum: Optional[int] = 2,
    eps: float = 1e-7
):
    x = torch.nan_to_num(x)
    dist = Independent(Categorical(logits=logits_p.permute(0, 2, 3, 1)), x.ndim - 2)
    log_prob = dist.log_prob(x.float())
    return log_prob
