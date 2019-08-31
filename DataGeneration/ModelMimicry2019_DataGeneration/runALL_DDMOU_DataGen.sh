#!/bin/bash

for RANDSEEDID in 1 6 9 13 19
do
	mkdir -p DDMOU_DataGeneration_$RANDSEEDID
	cp -r DDMOU_DataGeneration_TEMPLATE/* DDMOU_DataGeneration_$RANDSEEDID
	cd DDMOU_DataGeneration_$RANDSEEDID
	qsub run_job_DDMOU.sh $RANDSEEDID
	cd ..
done	
