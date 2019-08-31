#!/bin/bash

for RANDSEEDID in 1 6 9 13 19
do
	mkdir -p LCA_DataGeneration_$RANDSEEDID
	cp -r LCA_DataGeneration_TEMPLATE/* LCA_DataGeneration_$RANDSEEDID
	cd LCA_DataGeneration_$RANDSEEDID
	qsub run_job_LCA.sh $RANDSEEDID
	cd ..
done	
