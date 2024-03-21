# $t^3$ VAE - univariate and bivariate synthetic data analyses

This subfolder contains the Pytorch implementation of the synthetic data analyses in Section 4.1. 

## Univariate dataset

### model training

To reproduce the 1-dimensional synthetic data analysis, use the following command:

```
python3 univariate_run.py --dirname [directory name]
```

One can modify **univariate_run.py** to work with other settings. 

###  MMD test

Once the above training is done, use the following command to do the MMD test:

```
python3 univariate_test.py  --dirname [directory name]
```

It prints out the results of MMD test with full/right tail/left tail data. If the training process is insufficient, hypothesis test code would not run as the trained model cannot generate tail data. We recommend using a **epochs** argument greater than 20 for sufficient training. 


## Bivariate dataset

### model training

To reproduce the 2-dimensional synthetic data analysis, use the following command:

```
python3 bivariate_run.py --dirname [directory name]
```

One can modify **bivariate_run.py** to work with other settings. 

###  MMD test

Once the above training is done, use the following command to do the MMD test:

```
python3 bivariate_test.py  --dirname [directory name]
```

It prints out the results of MMD test with full/right tail/left tail data. If the training process is insufficient, hypothesis test code would not run as the trained model cannot generate tail data. We recommend using a **epochs** argument greater than 20 for sufficient training. 