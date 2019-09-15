# magnitude-sensitive-sequential-sampling-models
### Supplementary material

This repository provides additional material accompanying the manuscript *Comparison of magnitude-sensitive sequential sampling models in a simulation-based study*, including data files and computer code for fitting and data analysis. The manuscript is currently *under review*.

We included all procedures (written in Python 3 code) we used to generate simulation data and to fit models. Script files to run batch jobs on a cluster are included as well (when recomputing results, those will need to be adjusted depending on the computational facilities available).

#### Data generation:

All files can be found in the `DataGeneration/ModelMimicry2019_DataGeneration` folder. In this sub-directory the data is organised such that data generated by simulating diffusion-type models (pDDM, mDDM, mSOU, mUOU) are stored in folders starting with `DDMOU` and the data obtained from simulating the leaky-competing accumulator model can be found in the folders starting with `LCA`. Five different data sets were generated with different random seeds for each model. This is indicated by including either 1, 6, 9, 13 or 19 in the name of the sub-directory. This number is not the random seed but it is a factor used to compute the random seed. That is, random seed *factor 1* corresponds to *data set 1*, *factor 6* corresponds to *data set 2*, *factor 9* corresponds to *data set 3*, *factor 13* corresponds to *data set 4* and *factor 19* corresponds to *data set 5*. The factors were chosen arbitrarily.

Generated data is stored in csv-files and every folder containing data also includes a txt-file with more information about the data sets.
Simulated data can be visualised using the `Plot_MeanDT_ResponseProp_DataGen.ipynb` notebook located in the parent directory.

#### Model fitting:

All model fitting routines and results are located in the `ModelFitting` directory. Here we also include some information in the names of the sub-directories, such as the tolerance used in the fitting routine.

Fitted model parameters are stored in `csv-files` in the folder `CSV_Files`, which is located in the parent folder `ModelFitting`. In the `ModelFitting` directory we also included the notebook `GetFittedParameters.ipynb` which can be used to interactively retrieve fitted parameters and values of the objective function (BIC) by choosing `data set number` (1, 2, 3, 4 or 5) and `data-generating model` (pDDM, mDDM, mSOU, mUOU or LCA). This is explained in the notebook as well. 

#### Contact:

We hope this repository is useful for everyone interested in our study. In case you would like more information please get in touch (t.bose@sheffield.ac.uk).
