#!/bin/bash

FTOL=2000
XTOL=0.005
STRINGMODEL="LCA mDDM pDDM mSOU mUOU"
TMAX=6

for RANDSEEDID in 1 6 9 13 19
do
	mkdir -p ModelMimicry2019_PAPER_$RANDSEEDID
	cp -r ModelMimicry2019_TEMPLATE_PAPER/* ModelMimicry2019_PAPER_$RANDSEEDID
	cd ModelMimicry2019_PAPER_$RANDSEEDID
	for STR1 in $STRINGMODEL
	do
		cd Fit_to_${STR1}_data
		for STR2 in $STRINGMODEL
		do
			cd Fit_${STR2}_to_${STR1}
			qsub Run_${STR2}_fit_to_${STR1}_data.sh $RANDSEEDID $FTOL $XTOL $TMAX
			cd ..
		done
		cd ..
	done
	cd ..
done	
