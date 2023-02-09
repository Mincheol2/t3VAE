# γAE

This repository is the implementaion of our framework, made by **Jaehyuk Kwon**, **Juno Kim**, and **Mincheol Cho**.

## Abstract


## Basic Usage

### Baseline : Vanila VAE(with KL divergence)

- To train the baseline model,

```
python main.py --dataset mnist #Default KL Div
```

- If you want to change the default parameters(epoch, zdim, .. etc.), see the main.py.


### Experiments : γ-divergence.

- For reproducing our experiments, you may fine-tune these arguments.

|argument|description|default value|
|------|---|---|
|--beta|Weight for divergence loss. |1.0|
|--nu |Paramter for γ-divergence.|0.0 (unused)|
|--epochs| the number of epochs| 100 |

- If you test γ-divergence, please keep in mind **nu > 2**. (Because the variance of T distribution exists when nu > 2)

```
python main.py --dataset mnist --beta 1.0 --nu 3 #Gamma Div
```

