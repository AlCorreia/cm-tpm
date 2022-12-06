import numpy as np
import torch
import qmcpy


class GaussianQMCSampler:

    def __init__(
        self,
        latent_dim: int,
        n_bins: int,
        mean: float = 0.0,
        covariance: float = 1.0
    ):
        self.latent_dim = latent_dim
        self.n_bins = n_bins
        self.mean = mean
        self.covariance = covariance

    def __call__(
        self,
        seed: int = None,
        dtype=torch.float32
    ):
        if seed is None:
            seed = np.random.randint(1e9)
        discrete_dist = qmcpy.Lattice(dimension=self.latent_dim, randomize=True, seed=seed)
        true_measure = qmcpy.Gaussian(sampler=discrete_dist, mean=self.mean, covariance=self.covariance)
        z = torch.from_numpy(true_measure.gen_samples(self.n_bins))
        log_w = torch.full(size=(self.n_bins,), fill_value=np.log(1 / self.n_bins))
        return z.type(dtype), log_w.type(dtype)
