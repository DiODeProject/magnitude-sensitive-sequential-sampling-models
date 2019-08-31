import numpy.random as rand
import math
import numpy as np
import sys
from scipy.stats import chi2
from enum import Enum

if (len(sys.argv)>1):
	randSeedID = int(sys.argv[1])
else:
	randSeedID = 1

randomSeed=int(221189*randSeedID)
rand.seed(randomSeed)

#this file is produced taking into acount starting point variability and drift rate variability
filename = 'DDMOU-24Jun2019'

dt = 0.002 # integration step size for O-U and drift-diffusion processes
frequencyStimulusResample = 50
T_newFrame = int(1/(dt*frequencyStimulusResample))
count_t_increment = 0

sigma = 0.1 # standard deviation of Wiener process in generalised DDM
standard_dev_phys_stim = 0.1

def sampled_stimulus(val1, val2):
    v1 = val1 + rand.normal(0, standard_dev_phys_stim)
    if v1 < 0.1:
        v1 = 0.1
    elif v1 > 1.0:
        v1 = 1.0
    v2 = val2 + rand.normal(0, standard_dev_phys_stim)
    if v2 < 0.1:
        v2 = 0.1
    elif v2 > 1.0:
        v2 = 1.0
    return v1, v2

def stim_to_drift(v1,v2,gamma):
    drift = pow(v1, gamma) - pow(v2, gamma)
    return drift


### conditions index: 1:OU-stable, 2:OU-unstable, 3:DDM-multipl-noise, 4:DDM
conditions = [1, 2, 3, 4]

stimulus_value_pairs = [[0.4, 0.3], [0.6, 0.5], [0.6, 0.45], [0.45, 0.45]]

gammaArray = rand.uniform(0.3, 0.7, len(conditions))
thresholdArray = rand.uniform(0.1001, 0.4, len(conditions))
spvMaxArray = rand.uniform(0.05, 0.1, len(conditions))
drift_varArray = rand.uniform(0.04, 0.08, len(conditions))
decay = -rand.uniform(1.0, 4.0)
growth = rand.uniform(1.0, 4.0)
noiseCoefficientArray = rand.uniform(0.05, 0.2, len(conditions)-1)

trials = 20000

tMax = 10

##############################################################################################
# Predictor-corrector integration routine: Heun
##############################################################################################
# time t
# time step dt
# random numbers xi
# deterministic and stochastic contributons Fdet and Frand
def HeunRand(t, dt, y, Fdet, Frand, stdDev):
    yt = 0
    fd1 = Fdet(y)
    fr = Frand(stdDev)
    sqrtdt = math.sqrt(dt)
    yt = y + dt*fd1 + fr*sqrtdt
    fd2 = Fdet(yt)
    dt2 = dt/2e0
    y += dt2*(fd1+fd2) + fr*sqrtdt
    return y

##############################################################################################
# Euler integration routine for stochastic differential equations
##############################################################################################
def EulerRand(t, dt, y, Fdet, Frand, stdDev):
    fd = Fdet(y)
    fr = Frand(stdDev)
    return y + dt*fd + fr*math.sqrt(dt)

    
# deterministic part of RHS of dynamical system     
def Fdet(y):
    return A + B*y
   
# prefactor of stochastic part of RHS   
def Frand(stdDev):
    xi = rand.normal(0,1)
    return stdDev*xi
###############################################################################################

runID = 0
legend = open(filename + '-' + str(randSeedID) + '.txt', 'w')
rand.seed(randomSeed)
for condition in conditions: # conditions represent models
	gamma = gammaArray[condition-1]
	if condition == 4:
		z_lim = thresholdArray[condition-1]/2.0
		spvMax = spvMaxArray[condition-1]/2.0
	else:
		z_lim = thresholdArray[condition-1]
		spvMax = spvMaxArray[condition-1]
	drift_var = drift_varArray[condition-1]
	if condition < len(conditions):
		noiseCoef = noiseCoefficientArray[condition-1]
	for stimValPair in stimulus_value_pairs:
		runID += 1
		output = open(filename + '-' + str(randSeedID) + '-runid-' + str(runID) + '.csv', 'w')
		
		systemParam = 'modelClass: DDMOU randSeedID: ' + str(randSeedID) + ' runID: ' + str(runID) + ' trials: ' + str(trials) + ' stimulus value pair: ' + str(stimValPair) + ' threshold: ' + str(z_lim) + ' additive noise: ' + str(sigma) + ' gamma: ' + str(gamma) + ' SPV: ' + str(spvMax) + 'drift variability: ' + str(drift_var)
		# hack to set generalised DDM parameters based on condition
		if condition == 1:
			# stable O-U process with multiplicative noise (mSOU)
			B = decay
			systemParam = systemParam + ' B(decay): ' + str(B) + ' coeff_multiplicNoise: ' + str(noiseCoef) + ' model: mSOU'
		elif condition == 2:
			# unstable O-U process with multiplicative noise (mUOU)
			B = growth
			systemParam = systemParam + ' B(growth): ' + str(B) + ' coeff_multiplicNoise: ' + str(noiseCoef) + ' model: mUOU'
		elif condition == 3:
			# DDM with multiplicative noise (mDDM)
			B = 0
			systemParam = systemParam + ' B : ' + str(B) + ' coeff_multiplicNoise: ' + str(noiseCoef) + ' model: mDDM'
		else:
			# pDDM
			B = 0
			systemParam = systemParam + ' B: ' + str(B) + ' coeff_multiplicNoise: ' + str(0) + ' model: pDDM'
		
		for i1 in range(1, trials + 1):
			v1, v2 = sampled_stimulus(stimValPair[0], stimValPair[1])
			drift_var_trial = rand.normal(0, drift_var)
			A = stim_to_drift(v1,v2,gamma) + drift_var_trial
			if condition < len(conditions):
				variance_tot = (sigma*sigma) + noiseCoef*(pow(v1, 2*gamma) + pow(v2, 2*gamma))
				stdDev = math.sqrt(variance_tot)
			else:
				stdDev = sigma
			
			x = rand.uniform(-spvMax, spvMax)  # BIASED starting point for decision variable
			t = 0 # time
			while (abs(x) < abs(z_lim) and t < tMax):
				# decision criterion not yet reached
				x = EulerRand(t,dt,x,Fdet,Frand,stdDev)

				# increase time
				t += dt
				count_t_increment += 1 
				if count_t_increment == T_newFrame:
					v1, v2 = sampled_stimulus(stimValPair[0], stimValPair[1])
					A = stim_to_drift(v1,v2,gamma) + drift_var_trial	
					if condition < len(conditions):
						variance_tot = (sigma*sigma) + noiseCoef*(pow(v1, 2*gamma) + pow(v2, 2*gamma))
						stdDev = math.sqrt(variance_tot)
					count_t_increment = 0
			if x > 0:
				output.write(str(runID) + ",1," + str(t) + "\n")
			else:
				output.write(str(runID) + ",0," + str(t) + "\n")
		#print(systemParam)
		legend.write(systemParam + "\n")
		output.close()

legend.close()
