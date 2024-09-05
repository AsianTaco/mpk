# Multipole Kernel (MPK) convolutional neural networks

Convolutional neural network layer using symmetric kernels based on multipole expansion with spherical harmonics

Initial idea and implementation developed by [Tom Charnock et al. (2020)](https://doi.org/10.1093/mnras/staa682).
Ported to JAX and further test by [Simon Ding et al. (2024)](https://doi.org/10.48550/arXiv.2407.01391).
Generalised to multi-kernel layer embedding with LAX backend by [T. Lucas Makinen et al. (2024)](https://ui.adsabs.harvard.edu/link_gateway/2024arXiv240718909M/doi:10.48550/arXiv.2407.18909).

## Installation

Create the conda environment by running

```bash
conda create -n mpk 'python>=3.7'
```

Then install the repository as package via

```bash
conda activate mpk
cd mpk
pip install -e .
```

The main dependencies are jax and flax. If you like GPU support, please install the packages first according to ...
before running the pip install command.

You can test the installation via

## Use

## Testing

To execute automated test, please install the `pytest` package through
```bash
pip install mpk[test]
```
All automated tests can then be run by
```bash
pytest test
```