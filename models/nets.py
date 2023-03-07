import torch.nn as nn
import torch


def get_decoder_debd(
    latent_dim: int,
    out_features: int,
    n_layers: int,
    batch_norm: bool,
    final_act=None
):
    hidden_dims = torch.arange(latent_dim, out_features, (out_features - latent_dim) / n_layers, dtype=torch.int)
    decoder = nn.Sequential()
    for i in range(len(hidden_dims) - 1):
        decoder.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        decoder.append(nn.LeakyReLU())
        if batch_norm:
            decoder.append(nn.BatchNorm1d(hidden_dims[i + 1]))
    decoder.append(nn.Linear(hidden_dims[-1], out_features))
    if final_act is not None:
        decoder.append(final_act)
    return decoder


def get_encoder_debd(
    latent_dim: int,
    in_features: int,
    n_layers: int,
    batch_norm: bool
):
    enc_hidden_dim = torch.arange(
        latent_dim * 2, in_features, (in_features - latent_dim * 2) / n_layers, dtype=torch.int).flip(0)
    encoder = nn.Sequential()
    encoder.append(nn.Linear(in_features, enc_hidden_dim[0]))
    for i in range(len(enc_hidden_dim) - 1):
        encoder.append(nn.LeakyReLU())
        if batch_norm:
            encoder.append(nn.BatchNorm1d(enc_hidden_dim[i]))
        encoder.append(nn.Linear(enc_hidden_dim[i], enc_hidden_dim[i + 1]))
    return encoder


class ResBlockOord(nn.Module):
    """
    Residual block used in https://arxiv.org/abs/1711.00937.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


def mnist_conv_decoder(
    latent_dim: int,
    n_filters: int,
    batch_norm: bool = True,
    final_act: nn.Module = None,
    bias: bool = False,
    learn_std: bool = False,
    resblock: bool = False,
    out_channels: int = None
):
    nf = n_filters
    decoder = nn.Sequential()   
    decoder.append(nn.Unflatten(1, (latent_dim, 1, 1)))
    decoder.append(nn.ConvTranspose2d(latent_dim, nf * 4, 3, 2, 0, bias=bias))
    if batch_norm:
        decoder.append(nn.BatchNorm2d(nf * 4))
    decoder.append(nn.LeakyReLU())
    if resblock:
        decoder.append(ResBlockOord(nf * 4))

    # state size. (nf*8) x 4 x 4
    decoder.append(nn.ConvTranspose2d(nf * 4, nf * 2, 3, 2, 0, bias=bias))
    if batch_norm:
        decoder.append(nn.BatchNorm2d(nf * 2))
    decoder.append(nn.LeakyReLU())

    if resblock:
        decoder.append(ResBlockOord(nf * 2))

    # state size. (nf*4) x 8 x 8
    decoder.append(nn.ConvTranspose2d(nf * 2, nf, 3, 2, 0, bias=bias))
    if batch_norm:
        decoder.append(nn.BatchNorm2d(nf))
    decoder.append(nn.LeakyReLU())

    if resblock:
        decoder.append(ResBlockOord(nf))

    # state size. (nf*2) x 16 x 16
    if not out_channels:
        out_channels = 2 if learn_std else 1
    decoder.append(nn.ConvTranspose2d(nf, out_channels, 3, 2, 2, 1, bias=bias))

    if final_act is not None:
        decoder.append(final_act)

    return decoder


def svhn_conv_decoder(
    latent_dim: int,
    n_filters: int,
    batch_norm: bool = True,
    final_act: nn.Module = None,
    bias: bool = False,
    learn_std: bool = False,
    resblock: bool = False,
    out_channels: int = None
):
    nf = n_filters
    decoder = nn.Sequential()
    decoder.append(nn.Unflatten(1, (latent_dim, 1, 1)))
    decoder.append(nn.ConvTranspose2d(latent_dim, nf * 4, 4, 1, 0, bias=bias))
    if batch_norm:
        decoder.append(nn.BatchNorm2d(nf * 4))
    decoder.append(nn.LeakyReLU())
    if resblock:
        decoder.append(ResBlockOord(nf * 4))

    # state size. (nf*4) x 4 x 4
    decoder.append(nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1, bias=bias))
    if batch_norm:
        decoder.append(nn.BatchNorm2d(nf * 2))
    decoder.append(nn.LeakyReLU())

    if resblock:
        decoder.append(ResBlockOord(nf * 2))

    # state size. (nf*2) x 8 x 8
    decoder.append(nn.ConvTranspose2d(nf * 2, nf, 4, 2, 1, bias=bias))
    if batch_norm:
        decoder.append(nn.BatchNorm2d(nf))
    decoder.append(nn.LeakyReLU())

    if resblock:
        decoder.append(ResBlockOord(nf))

    # state size. (nf) x 16 x 16
    if not out_channels:
        out_channels = 6 if learn_std else 3
    decoder.append(nn.ConvTranspose2d(nf, out_channels, 4, 2, 1, bias=bias))

    if final_act is not None:
        decoder.append(final_act)

    return decoder
