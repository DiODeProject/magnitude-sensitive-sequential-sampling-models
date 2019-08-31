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
filename = 'LCA-24Jun2019'

dt = 0.002 # integration step size for O-U and drift-diffusion processes
frequencyStimulusResample = 50
T_newFrame = int(1/(dt*frequencyStimulusResample))
count_t_increment = 0

sigma_values = [0.1/math.sqrt(2)] # standard deviation of Wiener process in LCA
standard_dev_phys_stim = 0.1

gamma = rand.uniform(0.3, 0.7)
beta = rand.uniform(0.2, 2)
leak = rand.uniform(0.2, 2)

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
    return pow(v1, gamma), pow(v2, gamma)
 
stimulus_value_pairs = [[0.4, 0.3], [0.6, 0.5], [0.6, 0.45], [0.45, 0.45]]
threshSampled = rand.uniform(0.2001, 0.5)
thresholds = [threshSampled]
spvMax = rand.uniform(0.05, 0.2)

trials = 20000

### conditions index: 1:OU-stable, 2:OU-unstable, 3:DDM-multipl-noise, 4:DDM, 5:LCA
conditions = [5]
tMax = 10

#if (len(sys.argv)>1):
#    conditions=[int(sys.argv[1])]

runID=(conditions[0]-1)*len(stimulus_value_pairs)*len(thresholds)

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

def EulerRand(t, dt, y, Fdet, Frand, stdDev):
    fd = Fdet(y)
    fr = Frand(stdDev)
    yNew = y + dt*fd + fr*math.sqrt(dt)
    for jj in range(len(yNew)):
        yNew[jj] = max(0, yNew[jj])
    return yNew

#determnistic part
def Fdet(y):
    Fdet = np.zeros(2)
    Fdet[0] = -leak*y[0] - beta*y[1] + In0
    Fdet[1] = -leak*y[1] - beta*y[0] + In1
    return Fdet

# stochastic part of RHS   
def Frand(stdDev):
    Frand = np.zeros(2)
    xRand0 = rand.normal(0,1)
    xRand1 = rand.normal(0,1)
    Frand[0] = stdDev*xRand0
    Frand[1] = stdDev*xRand1
    return Frand
    
###############################################################################################

legend = open(filename + '-' + str(randSeedID) + '.txt', 'w')
for condition in conditions:
	for z in thresholds:
		z_lim = z
		for sigma in sigma_values:
			stdDev = sigma
			for stimValPair in stimulus_value_pairs:
				runID += 1
				output = open(filename + "-" + str(randSeedID) + "-runid-" + str(runID) + '.csv', 'w')

				systemParam = 'model: LCA randSeedID: ' + str(randSeedID) + ' runID: ' + str(runID) + ' trials: ' + str(trials) + ' threshold: ' + str(z_lim) + ' additive noise: ' + str(sigma) + ' gamma: ' + str(gamma)
				systemParam = systemParam + ' SPV: ' + str(spvMax) + ' leak: ' + str(leak) + ' beta: ' + str(beta) + ' stimulus value pair: ' + str(stimValPair)
				for i1 in range(1, trials + 1):
					In0, In1 = sampled_stimulus(stimValPair[0], stimValPair[1])
					x0init = rand.uniform(0,spvMax)
					x1init = rand.uniform(0,spvMax)
					x = np.array([x0init, x1init])  # starting point for decision variable
					t = 0 # time
					while (abs(x[0]) < abs(z_lim) and abs(x[1]) < abs(z_lim) and t < tMax):
						# decision criterion not yet reached
						x = EulerRand(t,dt,x,Fdet,Frand,stdDev)

						# increase time
						t += dt
						count_t_increment += 1 
						if count_t_increment == T_newFrame:
							In0, In1 = sampled_stimulus(stimValPair[0], stimValPair[1])
							count_t_increment = 0

					if x[0] > x[1]:
						output.write(str(runID) + ",1," + str(t) + "\n")
					else:
						output.write(str(runID) + ",0," + str(t) + "\n")

				#print(systemParam)
				legend.write(systemParam + "\n")
				output.close()

legend.close()


