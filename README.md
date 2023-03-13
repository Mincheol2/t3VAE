# γAE

Pytorch implementaion of γAE framework, made by **Jaehyuk Kwon**, **Juno Kim**, and **Mincheol Cho**.

## Abstract


## Basic Usage

### Run model

- This model covers three types of VAE frameworks : VAE, RVAE(Robust Variational AutoEncoder using beta divergence), and γAE(ours).

- The type of models can be determined by varying argument types. (nu, beta)


#### γAE 

- Use a 'nu' argument.

- Note that **γ = -2 / (p + q + nu)**, where p = data dimension, q = latent dimesnion.

- If you test γAE, please keep in mind **nu > 2**. (Because the variance of T distribution exists when nu > 2)


```
python main.py --dataset mnist --nu 2.5
```

#### RVAE

- Use a 'beta' argument. This is the hyperparameter for beta ELBO. For details, see [the original paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705121010534)

- Usually use beta = 0.005

```
python main.py --dataset mnist --beta 0.005
```

#### Vanila VAE

- Both types of arguments are not used.

```
python main.py --dataset mnist
```


### Arguments description.

- To reproducing our experiments, you may fine-tune these arguments.

|argument|description|Default|
|------|---|---|
|--dataset|Dataset type| mnist|
|--beta|RVAE hyperparatmer for beta ELBO| |
|--nu |γAE hyperparatmer for γ-divergence.||
|--epoch |Learning epochs |100|
|--train_frac |Proportion of contaminated data in the train dataset.(0~1)|0|
|--test_frac |Proportion of contaminated data in the test dataset.(0~1)|0|

- If you want to change other parameters(zdim, lr, .. etc.), see the argument.py.
