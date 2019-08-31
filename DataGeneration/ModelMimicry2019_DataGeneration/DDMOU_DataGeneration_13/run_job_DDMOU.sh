#!/bin/bash
#Request 2 gigabytes of virtual (mem) and real (rmem) memory
#$ -l mem=2G -l rmem=2G
#$ -N ouddm

#Load the Anaconda Python 3 Environment modulue
module load apps/python/anaconda3-2.5.0

#Run the hello.py program
python DDMOU_Generate_ExpData_2019.py $1
