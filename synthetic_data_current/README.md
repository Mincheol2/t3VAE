# $t^3$ VAE - 1-dimensionalsynthetic data analysis

This subfolder contains the Pytorch implementation of the synthetic data analyses in the section 4.1 of $t^3$ VAE paper. 

## model training

To reproduce the 1-dimensional synthetic data analysis, use the following command:

```
python3 univariate_run.py --dirname [directory name]
```

One can modify **univariate_run.py** to work with other settings. 

##  MMD test

Once the above training is done, one can do the MMD test by using the following command with the same arguments:

```
python3 univariate_test.py  --dirname [directory name]
```

It prints out the results of MMD test with full/right tail/left tail data. If the training process is insufficient, hypothesis test code would not run as the trained model cannot generate tail data. We recommend using a **epochs** argument greater than 20 for sufficient training. 

## Reproducibility

All default setups are equivalent to the reported data analysis. If GPU memory is enough, one can repeat the same experiment using the following commands : 

```
python3 univariate_run.py
python3 univariate_test.py 
```

or

```
./univariate_run.sh
```


# $t^3$ VAE - 2-dimensionalsynthetic data analysis

Similarly to above, one can reproduce the 2-dimensional synthetic data analysis

## model training

To reproduce the 2-dimensional synthetic data analysis, use the following command:

```
python3 bivariate_run.py --dirname [directory name]
```

One can modify **bivariate_run.py** to work with other settings. 

##  MMD test

Once the above training is done, one can do the MMD test by using the following command with the same arguments:

```
python3 bivariate_test.py  --dirname [directory name]
```

It prints out the results of MMD test with full/right tail/left tail data. If the training process is insufficient, hypothesis test code would not run as the trained model cannot generate tail data. We recommend using a **epochs** argument greater than 20 for sufficient training. 

## Reproducibility

All default setups are equivalent to the reported data analysis. If GPU memory is enough, one can repeat the same experiment using the following commands : 

```
python3 bivariate_run.py
python3 bivariate_test.py 
```

or

```
./bivariate_run.sh
```

