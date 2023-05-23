# $t^3$ VAE

Pytorch implementation of $t^3$ VAE framework.

## Basic Usage

This model covers multiple types of VAE framework, including $t^3$ VAE(ours), Guassian VAE, $\beta$-VAE , tilted VAE, and Factor VAE

The specific type of model can be determined by varying the argument "--model" or other hyperparamters(e.g. "--recon_sigma" or "--reg_weight".

To run the model, use the following command:

```
python run.py --model [model_type] --batch_size [batch_size] --epoch [epoch] --m_dim [latent_dim]
```

### $t^3$ VAE

- For the gamma-pow divergence, must use the "--nu" argument. Please keep in mind **$\nu$ > 2**.

- Note that **$\gamma = - \frac{2} {m + n + \nu}$**, where m = latent variable dimension and n = data dimension.

```
python run.py --model t3VAE --nu 2.5
```

### Other frameworks

- "--reg_weight" : a hyperparameter for $\beta$ -VAE.

- "--prior_sigma" : a hyperparamter for the variance of prior. 

- To use other framework such as tilted VAE and FactorVAE, change the model name in the command. 

```
python run.py --model VAE
python run.py --model VAE --prior_sigma 1.5 # larger variance
python run.py --model VAE --beta_weight 0.1 # betaVAE with beta =0.1
python run.py --model TiltedVAE --tilt 40
python run.py --model FactorVAE --TC_gamma 6.4
```

## Arguments description

|argument|description|Default|
|------|---|---|
|model|model type|VAE|
|datapath|dataset path||
|epoch|latent variable dimension|50|
|batch_size|latent variable dimension|64|
|lr| learning rate|1e-4|
|m_dim|latent variable dimension|64|
|nu|hyperparatmer for γ-divergence (Only used in $t^3$ VAE)||
|beta_weight|weight of the regularizer loss| 1.0|
|prior_sigma|standard deviation of the prior| 1.0|
|tilt|tilting parameter $\tau$ (Only used in TitledVAE)| 40|
|TC_gamma|TC regularizer weight (Only used in FactorVAE)| 6.4|
|lr_D|learning rate of the discriminator(Only used in FactorVAE)| 1e-5|
