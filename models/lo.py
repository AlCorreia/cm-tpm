from models.cm import ContinuousMixture
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch


def bins_lo(
    model: ContinuousMixture,
    n_bins: int,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    max_epochs: int = 100,
    lr: float = 0.01,
    patience: int = 10,
    progress_bar: bool = False,
    device: str = 'cuda'
):
    model.eval()
    model.sampler.n_bins = n_bins
    z_sampler, log_w = model.sampler(seed=42)
    log_w = log_w.to(device)

    z = nn.Embedding(num_embeddings=z_sampler.size(0), embedding_dim=z_sampler.size(1)).to(device)
    z.weight.data = z_sampler.to(device)
    optimizer = torch.optim.Adam(z.parameters(), lr=lr, weight_decay=1e-5)

    best_valid_loss = torch.inf
    early_stopping_counter = 0
    z_best = z.weight.data.detach().clone()
    epoch_iterator = tqdm(range(max_epochs)) if progress_bar else range(max_epochs)
    for _ in epoch_iterator:

        for x in train_loader:
            optimizer.zero_grad()
            log_prob = model(x.to(device), z.weight, log_w)
            train_loss = -log_prob.mean()
            train_loss.backward()
            optimizer.step()

        with torch.no_grad():
            valid_lls = []
            for x in valid_loader:
                log_prob = model(x.to(device), z.weight.data, log_w)
                valid_lls.append(-log_prob.mean().item())
            valid_loss_epoch = np.mean(valid_lls)

        if valid_loss_epoch >= best_valid_loss:
            early_stopping_counter += 1
            if early_stopping_counter == patience:
                break
        else:
            z_best = z.weight.data.detach().clone()
            early_stopping_counter = 0
            best_valid_loss = valid_loss_epoch

    return z_best, log_w


def fast_bins_lo(
    model: ContinuousMixture,
    n_bins: int,
    loader: DataLoader,
    n_epochs: int = 100,
    lr: float = 0.01,
    progress_bar: bool = False,
    device: str = 'cuda'
):
    model.eval()
    model.sampler.n_bins = n_bins
    z_sampler, log_w = model.sampler(seed=42)
    log_w = log_w.to(device)

    z = nn.Embedding(num_embeddings=z_sampler.size(0), embedding_dim=z_sampler.size(1)).to(device)
    z.weight.data = z_sampler.to(device)
    optimizer = torch.optim.Adam(z.parameters(), lr=lr, weight_decay=1e-5)

    epoch_iterator = tqdm(range(n_epochs)) if progress_bar else range(n_epochs)
    for _ in epoch_iterator:
        for x in loader:
            optimizer.zero_grad()
            log_prob = model(x.to(device), z.weight, log_w)
            loss = -log_prob.mean()
            loss.backward()
            optimizer.step()

    return z.weight.data, log_w
