from torch.utils.data import DataLoader
from torch.distributions import Normal
from typing import Callable, Optional
from utils.losses import bce_loss, mse_loss, ce_loss
import pytorch_lightning as pl
from tqdm import tqdm
import torch.nn as nn
import torch


class CLTBernoulliDecoder(nn.Module):

    def __init__(
        self,
        net: nn.Module,
        tree: list,
    ):
        super(CLTBernoulliDecoder, self).__init__()
        self.net = net
        self.tree = torch.Tensor(tree).type(torch.long)
        self.root = torch.where(self.tree == -1)[0].item()
        self.n_features = len(tree)
        self.features = torch.arange(self.n_features).type(torch.long)

    def _preprocess_logits_p(
        self,
        logits_p: torch.Tensor
    ):
        logits_p = logits_p.view(-1, self.n_features, 2)
        mask_1 = torch.ones_like(logits_p, dtype=torch.int8)
        mask_1[:, self.root, 0] = 0
        mask_2 = torch.zeros_like(logits_p)
        mask_2[:, self.root, 0] = logits_p[:, self.root, 1]
        logits_p = (logits_p * mask_1) + mask_2
        return logits_p

    def forward(
        self,
        x: torch.Tensor,
        log_w: torch.Tensor,
        z: torch.Tensor,
        k: Optional[int] = None,
        missing: Optional[bool] = None,
        n_chunks: Optional[int] = None
    ):
        assert missing is False, 'Marginalisation is not yet implemented!'
        assert x.ndim == 2, 'Only tabular data are allowed!'

        if k is not None:
            raise NotImplementedError

        logits_p = self._preprocess_logits_p(self.net(z))
        x_cond = torch.stack([1 - x[:, self.tree], x[:, self.tree]], dim=2)
        x_doubled = torch.cat([x, x], dim=1).view(-1, 2, self.n_features).permute(0, 2, 1)

        log_prob_chunks = []
        logits_p_chunks = tuple([logits_p]) if n_chunks is None else logits_p.chunk(n_chunks, dim=0)
        for logits_p_chunk in logits_p_chunks:
            lls = bce_loss(logits_p_chunk, x_doubled, dim_start_sum=None)
            log_prob_chunks.append(torch.einsum('bijk, bjk -> bij', lls, x_cond).sum(-1))
        log_prob_bins = torch.cat(log_prob_chunks, dim=1)
        return log_prob_bins


class BernoulliDecoder(nn.Module):

    def __init__(
        self,
        net: nn.Module
    ):
        super(BernoulliDecoder, self).__init__()
        self.net = net

    def forward(
        self,
        x: torch.Tensor,
        log_w: torch.Tensor,
        z: torch.Tensor,
        k: Optional[int] = None,
        missing: Optional[bool] = None,
        n_chunks: Optional[int] = None
    ):
        if k is not None:
            raise NotImplementedError

        logits_p = self.net(z)
        logits_p_chunks = tuple([logits_p]) if n_chunks is None else logits_p.chunk(n_chunks, dim=0)
        log_prob_bins = torch.cat([bce_loss(logits_p_chunk, x, missing) for logits_p_chunk in logits_p_chunks], dim=1)
        return log_prob_bins + log_w.unsqueeze(0)


class CategoricalDecoder(nn.Module):
    def __init__(
            self,
            net: nn.Module):
        super(CategoricalDecoder, self).__init__()
        self.net = net

    def forward(
            self,
            x: torch.Tensor,
            log_w: torch.Tensor,
            z: torch.Tensor,
            k: Optional[int] = None,
            missing: Optional[bool] = None,
            n_chunks: Optional[int] = None,
    ):
        z_chunks = tuple([z]) if n_chunks is None else z.chunk(n_chunks, dim=0)
        batch_size = x.shape[0]
        if k is not None:
            with torch.no_grad():
                # Run approximate posterior to find the 'best' k z values for each x
                log_prob_bins = torch.cat([ce_loss(self.net(z_chunk), x, k=None, missing=missing) for z_chunk in z_chunks],
                                          dim=1)
                log_prob_bins = log_prob_bins + log_w.unsqueeze(0)
                z_top_k = z[torch.topk(log_prob_bins, k=k, dim=-1)[1]]  # shape (batch_size, k, latent_dim)
            log_prob_bins_top_k = ce_loss(self.net(z_top_k.view(batch_size * k, -1)), x, k=k, missing=missing)
            return log_prob_bins_top_k
        else:
            log_prob_bins = torch.cat([ce_loss(self.net(z_chunk), x, missing) for z_chunk in z_chunks], dim=1)
            return log_prob_bins + log_w.unsqueeze(0)


class GaussianDecoder(nn.Module):

    def __init__(
        self,
        net,
        learn_std: bool = True,
        min_std: float = 0.1,
        max_std: float = 1.0,
        mu_activation=nn.Identity(),
    ):
        super(GaussianDecoder, self).__init__()
        self.net = net
        self.learn_std = learn_std
        self.min_std = min_std
        self.max_std = max_std
        self.mu_activation = mu_activation

    def forward(
        self,
        x: torch.Tensor,
        log_w: torch.Tensor,
        z: torch.Tensor,
        k: Optional[int] = None,
        missing: Optional[bool] = None,
        n_chunks: Optional[int] = None
    ):
        if k is not None:
            raise NotImplementedError

        if self.learn_std:
            mu_logvar = self.net(z)
            mu, logvar = mu_logvar.chunk(2, dim=1)
            mu = self.mu_activation(mu)
            std = torch.clamp(torch.exp(0.5 * logvar), min=self.min_std, max=self.max_std)
        else:
            mu = self.net(z)
            mu = self.mu_activation(mu)
            std = torch.full_like(mu, fill_value=self.min_std)
        mu_std_chunks = [[mu], [std]] if n_chunks is None else [[*mu.chunk(n_chunks, 0)], [*std.chunk(n_chunks, 0)]]
        log_prob = torch.cat(
            [mse_loss(mu_std_chunks[0][i], mu_std_chunks[1][i], x, missing) for i in range(len(mu_std_chunks[0]))], dim=1)
        return log_prob + log_w.unsqueeze(0)

    def sample_continuous(
        self,
        z: torch.Tensor,
        std_correction: float = 1.,
        device: str = 'cuda'
    ):
        if self.learn_std:
            mu_logvar = self.net(z.to(device))
            mu, logvar = mu_logvar.chunk(2, dim=1)
            mu = self.mu_activation(mu)
            std = torch.clamp(torch.exp(0.5 * logvar), min=self.min_std, max=self.max_std)
            samples = Normal(mu, std * std_correction).sample()
        else:
            mu = self.mu_activation(self.net(z.to(device)))
            samples = mu + torch.randn_like(mu) * self.min_std * std_correction
        return samples

    def sample_mixture(
        self,
        z: torch.Tensor,
        log_w: torch.Tensor,
        std_correction: float = 1.,
        device: str = 'cuda',
    ):
        if self.learn_std:
            raise Exception('Not implemented.')
        else:
            components = self.mu_activation(self.decoder.net(z.to(device)))
            idx = torch.distributions.Categorical(logits=log_w).sample([z.size(0)])
            samples = components[idx] + torch.randn_like(components[idx]) * self.decoder.min_std * std_correction
        return samples


class ContinuousMixture(pl.LightningModule):

    def __init__(
        self,
        decoder: nn.Module,
        sampler: Callable,
        k: Optional[int] = None
    ):
        super(ContinuousMixture, self).__init__()
        self.decoder = decoder
        self.sampler = sampler
        self.k = k
        self.n_chunks = None
        self.missing = None
        self.save_hyperparameters(ignore=['n_chunks', 'missing'])

    def forward(
        self,
        x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        log_w: Optional[torch.Tensor] = None,
        k: Optional[int] = None,
        seed: Optional[int] = None
    ):
        assert (z is None and log_w is None) or (z is not None and log_w is not None)
        if z is None:
            z, log_w = self.sampler(seed=seed)
        log_prob_bins = self.decoder.forward(x, log_w.to(x.device), z.to(x.device), k, self.missing, self.n_chunks)
        assert log_prob_bins.size() == (x.size(0), z.size(0))
        log_prob = torch.logsumexp(log_prob_bins, dim=1, keepdim=False)
        return log_prob

    def training_step(
        self,
        x: torch.Tensor,
        batch_idx: int
    ):
        log_prob = self.forward(x, k=self.k)
        loss = (-log_prob).mean()
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(
        self,
        x: torch.Tensor,
        batch_idx: int
    ):
        log_prob = self.forward(x, k=None, seed=42)
        loss = (-log_prob).mean()
        self.log('valid_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    @torch.no_grad()
    def eval_loader(
        self,
        loader: DataLoader,
        z: Optional[torch.Tensor] = None,
        log_w: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        progress_bar: Optional[bool] = False,
        device: str = 'cuda'
    ):
        self.eval()
        loader = tqdm(loader) if progress_bar else loader
        lls = torch.cat([self.forward(x.to(device), z, log_w, k=None, seed=seed) for x in loader], dim=0)
        assert len(lls) == len(loader.dataset)
        return lls

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
