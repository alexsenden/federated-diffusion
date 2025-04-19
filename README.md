# Federated Diffusion

### Overview

This repository is a set of three approaches to training diffusion models: one standard (centralized-model); one federated learning with a centralized aggregator (centralized-fl); and one federated learning with decentralized aggregation (decentralized-fl).

This repository also includes code to evaluate the trained models via IS, FID, a downstream ResNet50 classifier, and a t-SNE projection of latent vectors.

### Running

Each subfolder contains instructions to run that part of the project in its respective README, however they all require the global Python dependencies to be installed:

```
pip install -r requirements.txt
```