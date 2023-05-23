# t3VAE - synthetic data analysis

This subfolder contains the Pytorch implementation of the synthetic data analysis in the section 4.1 of t3VAE paper. 

## Model training and evaluation

To run the model, use the following command:

```
python3 run.py  --model_nu_list [model_nu]
```

where model_nu = 0 implies the Gaussian VAE. 

## MMD test

Once the above training is done, one can do the MMD test by using the following command:

```
python3 hypothesis_test.py  --dirname [directory name]
```

It prints out the results of MMD test with right/left tail data. Note that the results of MMD test with full data is already contained in the tensorboard in that directory. 

## Reproducibility

All default setups are equivalent to the reported data analysis. One can reproduce the same result by using the following command : 

```
python3 run.py
python3 hypothesis_test.py
```

or

```
run.sh
```