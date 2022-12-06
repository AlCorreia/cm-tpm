from models.nets import get_encoder_debd, get_decoder_debd
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np
import torch


class VAE(pl.LightningModule):

    def __init__(
        self,
        vae: nn.Module,
        recon_loss
    ):
        super(VAE, self).__init__()
        self.vae = vae
        self.recon_loss = recon_loss
        self.save_hyperparameters()

    def training_step(
        self,
        x: torch.Tensor,
        batch_idx: int,
    ):
        x_recon, _, mu, logvar = self(x)
        recon_loss, kld_loss = self.loss(x, x_recon, mu, logvar)
        loss = recon_loss + kld_loss

        # logging and return
        tensorboad_log = {
            'train_recon': recon_loss,
            'train_kld': kld_loss,
            'train_loss': loss,
        }
        self.log_dict(tensorboad_log, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(
        self,
        x: torch.Tensor,
        batch_idx: int,
    ):
        x_recon, _, mu, logvar = self(x)
        recon_loss, kld_loss = self.loss(x, x_recon, mu, logvar)
        loss = recon_loss + kld_loss

        # logging and return
        tensorboad_log = {
            'valid_recon': recon_loss,
            'valid_kld': kld_loss,
            'valid_loss': loss,
        }
        self.log_dict(tensorboad_log, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return loss

    def log_prob(
        self,
        x: torch.Tensor,
        n_mc_samples: int = 1,
        n_chunks: int = None,
    ):
        # Compute KL divergence
        mu, logvar = self.vae.encoder(x).split(self.vae.latent_dim, dim=1)
        kld = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=[i+1 for i in range(x.ndim-1)])
        # Compute reconstruction error with n_mc_samples
        mu = mu.repeat_interleave(n_mc_samples, 0)
        logvar = logvar.repeat_interleave(n_mc_samples, 0)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        x_recon = []
        z_chunks = tuple([z]) if n_chunks is None else z.split(int(z.size(0) / n_chunks), 0)
        for z_chunk in z_chunks:
            x_recon.append(self.vae.decoder(z_chunk))
        x_recon = torch.cat(x_recon, dim=0)

        all_recon = self.recon_loss(x_recon, x.repeat_interleave(n_mc_samples, 0))
        recon = all_recon.view(x.shape[0], n_mc_samples, -1).sum(dim=[i+2 for i in range(x.ndim-1)])
        # Compute log_prob and average over n_mc_samples
        log_prob = -(kld[..., None] + recon)
        log_prob = log_prob.logsumexp(dim=1) - np.log(n_mc_samples)
        return log_prob

    def forward(
        self,
        x: torch.Tensor
    ):
        mu, logvar = self.vae.encoder(x).split(self.vae.latent_dim, dim=1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        x_recon = self.vae.decoder(z)
        return x_recon, z, mu, logvar

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-5)

    def loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ):
        sum_dims = [i + 1 for i in range(x.ndim - 1)]
        recon_loss = self.recon_loss(x_recon, x).sum(dim=sum_dims).mean(dim=0)
        kld_loss = (-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=sum_dims)).mean(dim=0)
        return recon_loss, kld_loss

    def sample(
        self,
        n_samples: int,
        device: str = 'cuda'
    ):
        with torch.no_grad():
            z = torch.randn((n_samples, self.vae.latent_dim)).to(device)
            samples = self.decoder(z)
        return samples


class DebdVAE(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        n_features: int,
        n_layers: int = 6,
        batch_norm: bool = True,
    ):
        super(DebdVAE, self).__init__()
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.encoder = get_encoder_debd(latent_dim, n_features, n_layers, batch_norm)
        self.decoder = get_decoder_debd(latent_dim, n_features, n_layers, batch_norm, nn.Sigmoid())
