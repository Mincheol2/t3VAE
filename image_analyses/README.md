# $t^3$ VAE - Learning High-Dimensional Images

## Basic Usage

This repository includes multiple types of VAE framework:

$t^3$ VAE, Guassian VAE, $\beta$-VAE, Student-t VAE, DEVAE, tilted VAE, and Factor VAE

The specific type of model can be determined by varying the argument "--model" or other hyperparamters.(e.g. "--recon_sigma" or "--reg_weight")

To run the model, use the following command:

```
python run.py --model [model_type] --batch_size [batch_size] --epoch [epoch] --m_dim [latent_dim]
```

You can see example shell files on the **./exp_env**

### $t^3$ VAE

- For the gamma-pow divergence, you must use the "--nu" argument. Please keep in mind **$\nu$ > 2**.
- Note that **$\gamma = - \frac{2} {m + n + \nu}$**, where $m$ is *latent variable dimension* and $n$ is *data dimension*.

```
python run.py --model t3VAE --nu 2.5
```

### Hyperparameter for other frameworks

- "--reg_weight" : the weight of regularizer in $\beta$-VAE.
- "--prior_sigma" : the variance of prior.
- To use other framework such as tilted VAE and FactorVAE, change the model name in the command.

```
python run.py --model VAE
python run.py --model VAE --prior_sigma 1.5 # larger variance
python run.py --model VAE --beta_weight 0.1 # betaVAE with beta =0.1
python run.py --model TiltedVAE --tilt 40
python run.py --model FactorVAE --TC_gamma 6.4
```

## Arguments description


| argument    | description                                                | Default |
| ------------- | ------------------------------------------------------------ | --------- |
| model       | Model type                                                 | VAE     |
| datapath    | Dataset path                                               |         |
| epoch       | Latent variable dimension                                  | 50      |
| batch_size  | Latent variable dimension                                  | 64      |
| lr          | Learning rate                                              | 1e-4    |
| m_dim       | Latent variable dimension                                  | 64      |
| nu          | Hyperparatmer for γ-divergence (Only used in $t^3$ VAE)  |         |
| beta_weight | Weight of the regularizer loss                             | 1.0     |
| prior_sigma | Standard deviation of the prior                            | 1.0     |
| tilt        | Initial tilting parameter$\tau$ (Only used in tilted VAE)  | 40      |
| TC_gamma    | TC regularizer weight (Only used in FactorVAE)             | 6.4     |
| lr_D        | learning rate of the discriminator(Only used in FactorVAE) | 1e-5    |

## Evaluation

- You can evaluate trained model with **test.py** by computing sharpness and FID scores.

```
python test.py --model path [best_model path]
```
