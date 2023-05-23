# $t^3$ VAE

Pytorch implementation of $t^3$ VAE framework.

<img src="https://github.com/Mincheol2/t3VAE/assets/43122330/58b095d6-cc61-4896-bed1-cd489ecbf546" width="100%" height="100%">

## Abstract

The variational autoencoder (VAE) typically employs a standard normal prior as a regularizer for the probabilistic latent encoder. However, the Gaussian tail often decays too quickly to effectively accommodate the encoded points, failing to preserve crucial structures hidden in the data. In this paper, we explore the use of heavy-tailed models to combat over-regularization. Drawing upon insights from information geometry, we propose $t^3$ VAE, a modified VAE framework that incorporates Student's t-distributions for the prior, encoder, and decoder. This results in a joint model distribution of a power form which we argue can better fit real-world datasets. We derive a new objective by reformulating the evidence lower bound as joint optimization of a KL divergence between two statistical manifolds and replacing with $\gamma$-power divergence, a natural alternative for power families. $t^3$ VAE demonstrates superior generation of low-density regions when trained on heavy-tailed synthetic data. Furthermore, we show that our model outperforms other prior-based alternatives on CelebA and is able to encode and reconstruct various features in sharper detail.
