#!/bin/bash

#$ -l h_rt=96:00:00
# #$ -l arch=intel*
# #$ -l mem=4G
#$ -l rmem=2G
#$ -o Printout_params.txt

module load apps/python/anaconda3-4.2.0
python Fit_mSOU_to_mUOU.py $1 $2 $3 $4
module unload apps/python/anaconda3-4.2.0
