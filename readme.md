

Contains the code for variable selection in the linear regression model according to our article
"Disjunct Support Spike and Slab Priors for Variable Selection in Regression", Andrade et al., 2019

required python packages: numpy, scipy, rpy2

The r package imports in "imports.R" are only needed for the baselines.


In order to run the experiments for the proposed method on the synthethic data set use "runLinearRegressionSimData_ModelSearch.py".

python runLinearRegressionSimData_ModelSearch.py DATA_TYPE NOISE_RATIO porposed DELTA MCMC_SAMPLES

DATA_TYPE:
"correlated" corresponds to the low-dimensional simulated data.
"highDim" corresponds to the high-dimensional simulated data. 

NOISE_RATIO:
 0.0, 0.2, or 0.5
 
DELTA:
0.0, .., 0.8

MCMC_SAMPLES:
number of MCMC samples (10% from those samples are used for burn-in)


EXAMPLE:
/opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.0 proposed 0.5 10000



