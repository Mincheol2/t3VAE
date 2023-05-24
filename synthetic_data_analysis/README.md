# $t^3$ VAE - synthetic data analysis

This subfolder contains the Pytorch implementation of the synthetic data analysis in the section 4.1 of t3VAE paper. 

## Model training and evaluation

To run the model, use the following command:

```
python3 run.py --dirname [directory name] --model_nu_list [model_nu]
```

where model_nu = 0 implies the Gaussian VAE. 

## MMD test

Once the above training is done, one can do the MMD test by using the following command with the same arguments:

```
python3 hypothesis_test.py  --dirname [directory name] --model_nu_list [model_nu]
```

It prints out the results of MMD test with right/left tail data. Note that the results of MMD test with full data is already contained in the tensorboard. 

## Reproducibility

All default setups are equivalent to the reported data analysis. One can repeat the same experiment with following commands : 

```
python3 run.py
python3 hypothesis_test.py
```

or

```
run.sh
```