# $t^3$-VAE (working title)

The Pytorch implementation of the $t^3$-VAE(Triple-t) framework, developed by **Jaehyuk Kwon**, **Juno Kim**, and **Mincheol Cho**.

## Abstract


## Basic Usage

This model covers multiple types of VAE framework, including $t^3$ VAE(ours), Vanila VAE, , VampPrior, and TVAE.

The specific type of model can be determined by varying the argument "--model".

To run the model, use the following command:

```
python main.py  --model [model_type] --dataset [dataset_name]
```

### $t^3$-VAE

- For the gamma-pow divergence, use a "--nu" argument. Please keep in mind **$\nu$ > 2**.

- Note that **$\gamma = - \frac{2} {p + q + \nu}$**, where p = data dimension, q = latent variable dimesnion.


```
python main.py --model TtVAE --dataset celebA --nu 3
```

#### Other frameworks

- To use other framework, simply change the model name in the command. 

- For fine-tuning these models, refer to the argument description.
```
python main.py --model VAE --dataset celebA
```

### Arguments description

|argument|description|Default|
|------|---|---|
|dataset|Dataset type|celebA|
|nu |hyperparatmer for γ-divergence.||
|qdim| latent variable dimension| 64|
|reg_weight| weight of the regularizer loss| 1.0|
|recon_sigma| sigma value used in reconstruction term| 1.0|
|nums_component| number of pseudoinput components (Only used in VampPrior)|50|
