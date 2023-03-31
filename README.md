# $t^3$-VAE (working title)

Pytorch implementaion of $t^3$ VAE(Triple-t) framework, made by **Jaehyuk Kwon**, **Juno Kim**, and **Mincheol Cho**.

## Abstract


## Basic Usage

### Run Model

- This model covers N types of VAE framework : $t^3$ VAE(ours), Vanila VAE, , VampPrior, and TVAE
 
- The type of model can be determined by varying argument types. (--model)


#### t^3$-VAE

- Use a '--nu' argument. (because of the gamma-pow divergence)

- Note that **γ = $- \frac{2} {p + q + \nu}$**, where p = data dimension, q = latent variable dimesnion.

- When you test, please keep in mind **nu > 2**. (By definition)

```
python main.py --model TtAE --dataset celebA --nu 3
```

#### Other frameworks

- Just change model name. If you want to fine-tune these models, see the details on the arguments description.

```
python main.py --model VAE --dataset celebA
```

### Arguments description.

- To reproducing our experiments, you may fine-tune these arguments.

|argument|description|Default|
|------|---|---|
|dataset|Dataset type|celebA|
|nu |hyperparatmer for γ-divergence.||
|qdim| latent variable dimension| 64|
|reg_weight| weight of the regularizer loss| 1.0|
|recon_sigma| sigma value used in reconstruction term| 1.0|
|--nums_component| number of pseudoinput components (Only used in VampPrior)|50|

