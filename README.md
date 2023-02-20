# γAE

Pytorch implementaion of γAE framework, made by **Jaehyuk Kwon**, **Juno Kim**, and **Mincheol Cho**.

## Abstract


## Basic Usage

### Baseline : Vanila VAE(with KL divergence)

- To train the baseline model,

```
python main.py -dt ptb #Default KL Div
```

- If you want to change the default parameters(epoch, zdim, .. etc.), see the main.py.


### Experiments : γAE(γ-divergence)

- To reproducing our experiments, fine-tune these arguments.

|argument|description|default value|
|------|---|---|
|--beta|Weight for divergence loss. |1.0|
|--nu |Paramter for γ-divergence.|1.0|
|--epochs| the number of epochs| 100 |

- If you test γ-divergence, please keep in mind **nj > 2**. (Because the variance of T distribution exists when df > 2)

```
python main.py -dt ptb --beta 1.0 --df 3 #Gamma Div
```

