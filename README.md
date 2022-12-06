# Continuous Mixtures of Tractable Probabilistic Models

This repository is the official implementation of [Continuous Mixtures of Tractable Probabilistic Models](https://arxiv.org/abs/2209.10584) to appear at AAAI 2023. 

## Requirements

The code was developed and tested in Python 3.8. To install requirements:

```setup
pip install -r requirements.txt
```

## Training

All experiments are written as iPython notebooks and can be found in the `notebooks` folder.
Reproducing the experiments in the paper is as easy as running the cells in each notebook in order.

The notebooks are configured to run each experiment 5 times with random seeds in {0, 1, 2, 3, 4} to match the results in the paper. To change that it suffices to change the `seeds` variable in the notebook to something else.

The data used in all experiments is publicly available and automatically downloaded in the corresponding notebooks (see `dataset.py`).

We use [Pytorch Lightning](https://www.pytorchlightning.ai/) to manage training, so checkpoints are automatically saved `/logs/<dataset>/<model_type>/`.

### Continuous Mixtures
The following notebooks will train continuous mixtures on each of the datasets considered. 
The choice between a factorised or Chow-Liu structure can be made via the `use_clt` flag.
- `cm_debd_train.ipynb` trains continuous mixtures on the 20 density estimation benchmarks ([DEBD](https://github.com/arranger1044/DEBD)).
- `cm_bmnist_train.ipynb` trains continuous mixtures on the static binary-MNIST dataset.
- `cm_mnist_fashionmnist_train.ipynb` trains continuous mixtures on MNIST and Fashion-MNIST datasets.

### Competing methods
The notebooks below will train VAEs or standard mixture models on the DEBD datasets. For Einet experiments we refer to the [implemenation of Peharz et al.](https://github.com/cambridge-mlg/EinsumNetworks)
- `vae_debd_train.ipynb` trains standard VAEs on DEBD.
- `mixture_debd.ipynb` trains plain mixture models on DEBD.


## Evaluation
The trained models will be saved at `/logs/<dataset>/<model_type>/`. The following notebooks will search for trained models in those paths.

### Latent Optimisation
Once a model is trained, we can search for good integration points via latent optimisation. The notebooks below do so on the DEBD and binary MNIST datasets.
- `cm_debd_lo.ipynb` optimises integration points for trained continuous mixtures trained on DEBD.
- `cm_bmnist_lo.ipynb` optimises integration points for trained continuous mixtures trained on binary MNIST.

### Testing
The following notebooks evaluate trained models on the corresponding test data.
- `cm_debd_test.ipynb`
- `cm_bmnist_test.ipynb`
- `cm_mnist_fashionmnist_test.ipynb`
- `vae_debd_test.ipynb`

For each dataset and model type, the notebook will evaluate all models saved in the corresponding path `/logs/<dataset>/<model_type>/` and report the mean and stardard deviation over the different models (typically ran with different random seeds).

Test results still depend on RQMC sequences, which are stochastic. To reproduce the results in the paper, keep the random seed set to 42.
