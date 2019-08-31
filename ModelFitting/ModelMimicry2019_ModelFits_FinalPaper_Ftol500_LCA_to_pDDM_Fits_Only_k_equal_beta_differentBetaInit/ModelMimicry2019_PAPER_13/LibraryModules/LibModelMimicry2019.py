import numpy as np
import numpy.random as rand
from scipy.optimize import minimize
import math
import sys
import os
import copy

nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
	sys.path.append(nb_dir)

def getDataStats(chooseDataSetType, dataFilePath, runIDgroup, chooseRunIDgroup, stimulus_value_pairs, TMAXdec):
	bin_edges_allConds_Opt1 = []
	bin_edges_allConds_Opt2 = []
	abs_freq_allConds_Opt1 = []
	abs_freq_allConds_Opt2 = []
	for nn in runIDgroup[chooseRunIDgroup]:
		choice = []
		DT = []
		with open(str(dataFilePath) + str(nn) + '.csv', 'r') as infile:
			infile.readline
			for line in infile:
				if not line.startswith('#'): # skip lines starting with "#" 
					#runID.append(float(line.split(',')[0]))  # "split(',')" separates all objects in a line when delimeter "," occurs
					choice.append(float(line.split(',')[1]))
					DT.append(float(line.split(',')[2]))

		DTopt1 = np.asarray([DT[kk] for kk in range(len(DT)) if choice[kk]==1 and DT[kk]<=TMAXdec])
		DTopt2 = np.asarray([DT[kk] for kk in range(len(DT)) if choice[kk]==0 and DT[kk]<=TMAXdec])

		perc10_DTopt1 = np.percentile(DTopt1, 10, axis=0, interpolation='linear')
		perc30_DTopt1 = np.percentile(DTopt1, 30, axis=0, interpolation='linear')
		perc50_DTopt1 = np.percentile(DTopt1, 50, axis=0, interpolation='linear')
		perc70_DTopt1 = np.percentile(DTopt1, 70, axis=0, interpolation='linear')
		perc90_DTopt1 = np.percentile(DTopt1, 90, axis=0, interpolation='linear')
		perc10_DTopt2 = np.percentile(DTopt2, 10, axis=0, interpolation='linear')
		perc30_DTopt2 = np.percentile(DTopt2, 30, axis=0, interpolation='linear')
		perc50_DTopt2 = np.percentile(DTopt2, 50, axis=0, interpolation='linear')
		perc70_DTopt2 = np.percentile(DTopt2, 70, axis=0, interpolation='linear')
		perc90_DTopt2 = np.percentile(DTopt2, 90, axis=0, interpolation='linear')

		bin_edges_Opt1 = np.array([0, perc10_DTopt1, perc30_DTopt1, perc50_DTopt1, perc70_DTopt1, perc90_DTopt1, TMAXdec])
		bin_edges_Opt2 = np.array([0, perc10_DTopt2, perc30_DTopt2, perc50_DTopt2, perc70_DTopt2, perc90_DTopt2, TMAXdec])
		bin_edges_allConds_Opt1.append(bin_edges_Opt1)
		bin_edges_allConds_Opt2.append(bin_edges_Opt2)

		NrOpt1_arr, binsOpt1 = np.histogram(DTopt1, bins=bin_edges_Opt1)
		NrOpt2_arr, binsOpt2 = np.histogram(DTopt2, bins=bin_edges_Opt2)
		abs_freq_allConds_Opt1.append(NrOpt1_arr)
		abs_freq_allConds_Opt2.append(NrOpt2_arr)

	return bin_edges_allConds_Opt1, abs_freq_allConds_Opt1, bin_edges_allConds_Opt2, abs_freq_allConds_Opt2



def NeuroModel(Ntrials, stimValPair, fitParams, fixedParams, fittedModel, TMAXdec):

	z_lim = fitParams['threshold']
	if 'sigma' in fitParams.keys():
		sigma = fitParams['sigma']
	elif 'sigmaInit' in fixedParams.keys():
		sigma = fixedParams['sigmaInit']
	else:
		print("Error: check additive noise strength")
	gamma = fitParams['gamma']
	SPV = fitParams['SPV']
	if fittedModel != 'LCA':
		drift_var = fitParams['driftVar']
	else:
		beta = fitParams['beta']
		leak = beta

	if fittedModel in ['mDDM', 'mSOU', 'mUOU']:
		noiseCoef = fitParams['noiseCoeff']
	else:
		noiseCoef = None

	if fittedModel == 'mSOU':
		decay = fitParams['decay']
	elif fittedModel == 'mUOU':
		growth = fitParams['growth']


	if fittedModel == 'LCA':
		#Euler method
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
			Fdet[0] = -leak*y[0] - beta*y[1] + In0 # here leak=beta
			Fdet[1] = -leak*y[1] - beta*y[0] + In1 # here leak=beta
			return Fdet

		# stochastic part of RHS   
		def Frand(stdDev):
			Frand = np.zeros(2)
			xRand0 = rand.normal(0,1)
			xRand1 = rand.normal(0,1)
			Frand[0] = stdDev*xRand0
			Frand[1] = stdDev*xRand1
			return Frand
		# stimulus
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

	else:
		#Euler method   
		def EulerRand(t, dt, y, Fdet, Frand, stdDev):
			fd = Fdet(y)
			fr = Frand(stdDev)
			return y + dt*fd + fr*math.sqrt(dt)
		#deterministic part   
		def Fdet(y):
			return A + B*y
		#stochastic part
		def Frand(stdDev):
			xi = rand.normal(0,1)
			return stdDev*xi
		#stimulus
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
		# stimulus to drift conversion for diffusion-type models
		def stim_to_drift(v1,v2,gamma):
			drift = pow(v1, gamma) - pow(v2, gamma)
			return drift

	###############################################################################################
	dt = 0.002 # integration step size
	T_newFrame = 20
	count_t_increment = 0
	standard_dev_phys_stim = 0.1

	tMax = TMAXdec

	DTsimOpt1 = []
	DTsimOpt2 = []

	# if fittedModel = LCA process
	if fittedModel == 'LCA':
		stdDev = sigma
		for i1 in range(1, Ntrials + 1):
			In0, In1 = sampled_stimulus(stimValPair[0], stimValPair[1])
			x0init = rand.uniform(0,SPV)
			x1init = rand.uniform(0,SPV)
			x = np.array([x0init, x1init])

			t = 0 # time
			count_t_increment = 0

			while (abs(x[0]) < abs(z_lim) and abs(x[1]) < abs(z_lim) and t <= tMax):
				# decision criterion not yet reached
				x = EulerRand(t,dt,x,Fdet,Frand,stdDev)

				# increase time
				t += dt
				count_t_increment += 1
				if count_t_increment == T_newFrame:
					In0, In1 = sampled_stimulus(stimValPair[0], stimValPair[1])
					count_t_increment = 0

			if t <= tMax:
				if x[0] > x[1]:
					DTsimOpt1.append(t)
				else:
					DTsimOpt2.append(t)
	# if fittedModel in ['mDDM', 'pDDM', 'mSOU', 'mUOU']
	else:
		for i1 in range(1, Ntrials + 1):
			v1, v2 = sampled_stimulus(stimValPair[0], stimValPair[1])
			drift_var_trial = rand.normal(0,drift_var)
			A = stim_to_drift(v1,v2,gamma) + drift_var_trial

			if fittedModel == 'mDDM' or fittedModel == 'pDDM':
				B = 0
			elif fittedModel == 'mSOU':
				B = decay
			elif fittedModel == 'mUOU':
				B = growth

			# if multiplicative noise (mDDM or mSOU or mUOU)
			if noiseCoef:
				variance_tot = (sigma*sigma) + noiseCoef*(pow(v1, 2*gamma) + pow(v2, 2*gamma))
				stdDev = math.sqrt(variance_tot)
			else:
				stdDev = sigma

			x = rand.uniform(-SPV,SPV)  # BIASED starting point for decision variable
			t = 0 # time
			while (abs(x) < abs(z_lim) and t <= tMax):
				# decision criterion not yet reached
				x = EulerRand(t,dt,x,Fdet,Frand,stdDev)
				# increase time
				t += dt
				count_t_increment += 1
				if count_t_increment == T_newFrame:
					v1, v2 = sampled_stimulus(stimValPair[0], stimValPair[1])
					A = stim_to_drift(v1,v2,gamma) + drift_var_trial
					if noiseCoef:
						variance_tot = (sigma*sigma) + noiseCoef*(pow(v1, 2*gamma) + pow(v2, 2*gamma))
						stdDev = math.sqrt(variance_tot)
					count_t_increment = 0

			if t <= tMax:
				if x > 0:
					DTsimOpt1.append(t)
				else:
					DTsimOpt2.append(t)

	DTsimOpt1 = np.asarray(DTsimOpt1)
	DTsimOpt2 = np.asarray(DTsimOpt2)

	return DTsimOpt1, DTsimOpt2



def Fobj_AIC_BIC(varParVal, varParNames, fixPar, fittedModel, NumTrials, TMAXdec, stimulus_value_pairs, cond_list, 
		bin_edges_allConds_Opt1, abs_freq_allConds_Opt1, bin_edges_allConds_Opt2, abs_freq_allConds_Opt2,
		whichScore='BIC'):
	# par is vector with the parameters to varied as its components
	fitParams = {}
	for name, val in zip(varParNames, varParVal):
		if str(name) == 'decay' or str(name) == 'decay_in':
			fitParams[str(name)] = -abs(val)
		else:
			fitParams[str(name)] = abs(val)

	fixedParams = fixPar

	Likelihood_Opt1 = 0
	Likelihood_Opt2 = 0
	total_Nr_trials = np.sum(abs_freq_allConds_Opt1)+np.sum(abs_freq_allConds_Opt2) #experimental observations

	for jj in range(len(stimulus_value_pairs)):
		DTsimOpt1, DTsimOpt2 = NeuroModel(NumTrials, stimulus_value_pairs[jj], fitParams, fixedParams, fittedModel, TMAXdec)
		if len(DTsimOpt1)==0 or len(DTsimOpt2)==0:
			return 1e8

		print('meanDTopt1['+str(cond_list[jj])+'] = ', np.mean(DTsimOpt1))
		print('meanDTopt2['+str(cond_list[jj])+'] = ', np.mean(DTsimOpt2))
		NrOpt1_arr, binsOpt1 = np.histogram(DTsimOpt1, bins=bin_edges_allConds_Opt1[jj])
		NrOpt2_arr, binsOpt2 = np.histogram(DTsimOpt2, bins=bin_edges_allConds_Opt2[jj])
		NrOpt1plusOpt2 = np.sum(NrOpt1_arr) + np.sum(NrOpt2_arr)

		Prob_bins_Opt1 = NrOpt1_arr/NrOpt1plusOpt2
		Prob_bins_Opt1[Prob_bins_Opt1 < 1/NrOpt1plusOpt2] = 1/(10*NrOpt1plusOpt2)
		Likelihood_Opt1 += -2*np.sum(abs_freq_allConds_Opt1[jj]*np.log(Prob_bins_Opt1))

		Prob_bins_Opt2 = NrOpt2_arr/NrOpt1plusOpt2
		Prob_bins_Opt2[Prob_bins_Opt2 < 1/NrOpt1plusOpt2] = 1/(10*NrOpt1plusOpt2)
		Likelihood_Opt2 += -2*np.sum(abs_freq_allConds_Opt2[jj]*np.log(Prob_bins_Opt2))

	Likelihood = Likelihood_Opt1 + Likelihood_Opt2
	if whichScore == 'BIC':
		SCORE = Likelihood + len(fitParams)*math.log(total_Nr_trials)
	elif whichScore == 'AIC':
		SCORE = Likelihood + 2*len(fitParams) 
	else:
		print('ERROR: choose either AIC or BIC')
		return None
	print("Score = " + str(whichScore) + " = " + str(SCORE))
	return SCORE



def func_find_init_params(param_var_dict, param_fixed_dict, fittedModel, NumTrials, TMAXdec,
		stimulus_value_pairs, cond_list,
		bin_edges_allConds_Opt1, abs_freq_allConds_Opt1,
		bin_edges_allConds_Opt2, abs_freq_allConds_Opt2,
		scale=0.95, iterations=40, keepInd=15, nrChildren=3):
	pFixedDict = param_fixed_dict
	numberParamSets=keepInd*nrChildren
	fListNew = []
	pListsNew = []
	param_list = []
	param_names = []
	if 'threshInit' in param_var_dict.keys():
		param_list.append(param_var_dict['threshInit'])
		param_names.append('threshold')
	if 'gammaInit' in param_var_dict.keys():
		param_list.append(param_var_dict['gammaInit'])
		param_names.append('gamma')
	if 'spvInit' in param_var_dict.keys():
		param_list.append(param_var_dict['spvInit'])
		param_names.append('SPV')
	if 'driftVarInit' in param_var_dict.keys():
		param_list.append(param_var_dict['driftVarInit'])
		param_names.append('driftVar')
	if 'noiseCoeffInit' in param_var_dict.keys():
		param_list.append(param_var_dict['noiseCoeffInit'])
		param_names.append('noiseCoeff')
	if 'decayInit' in param_var_dict.keys():
		param_list.append(param_var_dict['decayInit'])
		param_names.append('decay')
	if 'growthInit' in param_var_dict.keys():
		param_list.append(param_var_dict['growthInit'])
		param_names.append('growth')
	if 'leakInit' in param_var_dict.keys():
		param_list.append(param_var_dict['leakInit'])
		param_names.append('leak')
	if 'betaInit' in param_var_dict.keys():
		param_list.append(param_var_dict['betaInit'])
		param_names.append('beta')
	if 'sigmaInit' in param_var_dict.keys():
		if param_var_dict['sigmaInit'] != None:
			param_list.append(param_var_dict['sigmaInit'])
			param_names.append('sigma')

	for it in range(iterations):
		pLists = copy.deepcopy(pListsNew)
		fList = copy.deepcopy(fListNew)
		if it == 0:
			nrNewParamSets = numberParamSets
			nrNewChildren = numberParamSets
			pLists.append(param_list)
		else:
			nrNewParamSets = keepInd
			nrNewChildren = nrChildren
		for nr in range(nrNewParamSets):
			paramList = pLists[nr]
			pmin_list = [paramList[kk] - paramList[kk]*pow(scale,it) for kk in range(len(paramList))]
			pmax_list = [paramList[kk] + paramList[kk]*pow(scale,it) for kk in range(len(paramList))]
			for jj in range(nrNewChildren):
				pList = [np.random.uniform(pmin_list[kk], pmax_list[kk]) for kk in range(len(paramList))]
				pLists.append(pList)
				#pNewDict = {}
				#for name, value in zip(param_names, pList):
				#    pNewDict[str(name)] = value    
				fobj = Fobj_AIC_BIC(pList, param_names, pFixedDict, fittedModel, NumTrials, TMAXdec,
						stimulus_value_pairs, cond_list,
						bin_edges_allConds_Opt1, abs_freq_allConds_Opt1,
						bin_edges_allConds_Opt2, abs_freq_allConds_Opt2)
				fList.append(fobj)
				fList, pLists = zip(*sorted(zip(fList, pLists)))
				fList = list(fList)
				pLists = list(pLists)
				fListNew = copy.deepcopy(fList[0:keepInd])
				pListsNew = copy.deepcopy(pLists[0:keepInd])
				#print(pListsNew[0:3])
				#print(fListNew[0:3])
	return pListsNew[0:3], param_names, fListNew[0:3]


def limitToPosNegValues(param_names, param_values):
	if type(param_values) != type([]):
		param_values = list(param_values)
	newVals = []
	for name, val in zip(param_names, param_values):
		if str(name) == 'decay' or str(name) == 'decay_in':
			newVals.append(-abs(val))
		else:
			newVals.append(abs(val))
	return np.asarray(newVals)

def printPart1A():
	print('###################################################################################################################')
	print('###################################################################################################################')
	print('################################ PART I - GENETIC ALGORITHM TO GET INITIAL PARAMETERS #############################')
	print('###################################################################################################################')
	print('###################################################################################################################')

def printPart1B(L_minStart, xStart, parameterListString, param_names, fittedModel):
	with open('OutFittedModParamGetInit.csv', 'w') as out:
		out.write('# SCORE, ' + str(parameterListString) + '\n')
		if 'sigma' not in param_names:
			if fittedModel == 'pDDM' or fittedModel == 'LCA':
				out.write(("{0:10.5f}{1:10.5f}{2:10.5f}{3:10.5f}{4:10.5f}").format(L_minStart, xStart[0], xStart[1], xStart[2], xStart[3]))
			elif fittedModel == 'mDDM':
				out.write(("{0:10.5f}{1:10.5f}{2:10.5f}{3:10.5f}{4:10.5f}{5:10.5f}").format(L_minStart, xStart[0], xStart[1], xStart[2], xStart[3], xStart[4]))
			elif fittedModel == 'mUOU' or fittedModel == 'mSOU':
				out.write(("{0:10.5f}{1:10.5f}{2:10.5f}{3:10.5f}{4:10.5f}{5:10.5f}{6:10.5f}").format(L_minStart, xStart[0], xStart[1], xStart[2], xStart[3], xStart[4], xStart[5]))
			else:
				print('Check fitted model!')
		else:
			if fittedModel == 'pDDM' or fittedModel == 'LCA':
				out.write(("{0:10.5f}{1:10.5f}{2:10.5f}{3:10.5f}{4:10.5f}{5:10.5f}").format(L_minStart, xStart[0], xStart[1], xStart[2], xStart[3], xStart[4]))
			elif fittedModel == 'mDDM':
				out.write(("{0:10.5f}{1:10.5f}{2:10.5f}{3:10.5f}{4:10.5f}{5:10.5f}{6:10.5f}").format(L_minStart, xStart[0], xStart[1], xStart[2], xStart[3], xStart[4], xStart[5]))
			elif fittedModel == 'mUOU' or fittedModel == 'mSOU':
				out.write(("{0:10.5f}{1:10.5f}{2:10.5f}{3:10.5f}{4:10.5f}{5:10.5f}{6:10.5f}{7:10.5f}").format(L_minStart, xStart[0], xStart[1], xStart[2], xStart[3], xStart[4], xStart[5], xStart[6]))
			else:
				print('Check fitted model!')



def printPart1C(time_GetInit_stop, time_GetInit_start, parameterListString, NtrialsGetInitBest3, xStart, L_minStart, NtrialsGetInitParams, fInitList, fValList, xStartList):
	print('########################################################################################################')
	print('elapsed time =',time_GetInit_stop - time_GetInit_start,'seconds')
	print('########################################################################################################')
	print('xStart = ' + str(parameterListString))
	print('xStart (' + str(NtrialsGetInitBest3) + ' trials) = ', xStart)
	print('Min SCORE (' + str(NtrialsGetInitBest3) + ' trials) = ', L_minStart)
	print('########################################################################################################')
	print('Min SCORE (list of best 3 with' + str(NtrialsGetInitParams) + ' trials) = ', fInitList)
	print('Min SCORE (list of best 3 with' + str(NtrialsGetInitBest3) + ' trials) = ', fValList)
	print('list of best 3 parameter sets xStart (' + str(NtrialsGetInitBest3) + ' trials) = ', xStartList)


def printPart2A():
	print('###################################################################################################################')
	print('###################################################################################################################')
	print('################################ PART II - MAIN FITTING ROUTINE (DOWNHILL-SIMPLEX) ################################')
	print('###################################################################################################################')
	print('###################################################################################################################')


def printPart2B(rep, NtrialsSimplex, parameterListString, xStart, xFinal, FitResFun, FitResMessage, FitResStatus, FitResNit, FitResNfev):
	print('########################################################################################################')
	print('########################################################################################################')
	print('Repetition number: ', str(1+rep))
	print('########################################################################################################')
	print('Number of trials in Downhill-Simplex = ' + str(NtrialsSimplex))
	print('xStart = ' + str(parameterListString))
	print('xStart = ', xStart)
	print('xFinal =', xFinal)
	print('SCORE = ', FitResFun)
	print('Fit message: ', FitResMessage)
	print('Status: ', FitResStatus, '(0: successful, 1: unsuccessful)')
	print('Iterations: ', FitResNit)
	print('Function evaluations: ', FitResNfev)


def printPart2C(time_Fit_stop, time_Fit_start):
	print('########################################################################################################')
	print('elapsed time =',time_Fit_stop - time_Fit_start,'seconds')
	print('########################################################################################################')


def printPart3A(scoreFinalMean, scoreFinalSTD, xMean, xSTD, param_names, fittedModel, parameterListString):
	print('###################################################################################################################')
	print('###################################################################################################################')
	print('################################ PART III - Create DataSet with Average Parameters ################################')
	print('###################################################################################################################')
	print('###################################################################################################################')
	print('scoreFinalMean =', scoreFinalMean)
	print('scoreFinalSTD =', scoreFinalSTD)
	print('Final Parameters Mean =', xMean)
	print('Final Parameters STD =', xSTD)

	with open('OutParamsFinalMeanStd.csv', 'w') as out2:
		out2.write('# SCORE, ' + str(parameterListString) + '\n')
		if 'sigma' not in param_names:
			if fittedModel == 'pDDM' or fittedModel == 'LCA':
				out2.write(("{0:10.5f}{1:10.5f}{2:10.5f}{3:10.5f}{4:10.5f}\n").format(scoreFinalMean, xMean[0], xMean[1], xMean[2], xMean[3]))
				out2.write(("{0:10.5f}{1:10.5f}{2:10.5f}{3:10.5f}{4:10.5f}\n").format(scoreFinalSTD, xSTD[0], xSTD[1], xSTD[2], xSTD[3]))
			elif fittedModel == 'mDDM':
				out2.write(("{0:10.5f}{1:10.5f}{2:10.5f}{3:10.5f}{4:10.5f}{5:10.5f}\n").format(scoreFinalMean, xMean[0], xMean[1], xMean[2], xMean[3], xMean[4]))
				out2.write(("{0:10.5f}{1:10.5f}{2:10.5f}{3:10.5f}{4:10.5f}{5:10.5f}\n").format(scoreFinalSTD, xSTD[0], xSTD[1], xSTD[2], xSTD[3], xSTD[4]))
			elif fittedModel == 'mUOU' or fittedModel == 'mSOU':
				out2.write(("{0:10.5f}{1:10.5f}{2:10.5f}{3:10.5f}{4:10.5f}{5:10.5f}{6:10.5f}\n").format(scoreFinalMean, xMean[0], xMean[1], xMean[2], xMean[3], xMean[4], xMean[5]))
				out2.write(("{0:10.5f}{1:10.5f}{2:10.5f}{3:10.5f}{4:10.5f}{5:10.5f}{6:10.5f}\n").format(scoreFinalSTD, xSTD[0], xSTD[1], xSTD[2], xSTD[3], xSTD[4], xSTD[5]))
			else:
				print('Check fitted model!')
		else:
			if fittedModel == 'pDDM' or fittedModel == 'LCA':
				out2.write(("{0:10.5f}{1:10.5f}{2:10.5f}{3:10.5f}{4:10.5f}{5:10.5f}\n").format(scoreFinalMean, xMean[0], xMean[1], xMean[2], xMean[3], xMean[4]))
				out2.write(("{0:10.5f}{1:10.5f}{2:10.5f}{3:10.5f}{4:10.5f}{5:10.5f}\n").format(scoreFinalSTD, xSTD[0], xSTD[1], xSTD[2], xSTD[3], xSTD[4]))
			elif fittedModel == 'mDDM':
				out2.write(("{0:10.5f}{1:10.5f}{2:10.5f}{3:10.5f}{4:10.5f}{5:10.5f}{6:10.5f}\n").format(scoreFinalMean, xMean[0], xMean[1], xMean[2], xMean[3], xMean[4], xMean[5]))
				out2.write(("{0:10.5f}{1:10.5f}{2:10.5f}{3:10.5f}{4:10.5f}{5:10.5f}{6:10.5f}\n").format(scoreFinalSTD, xSTD[0], xSTD[1], xSTD[2], xSTD[3], xSTD[4], xSTD[5]))
			elif fittedModel == 'mUOU' or fittedModel == 'mSOU':
				out2.write(("{0:10.5f}{1:10.5f}{2:10.5f}{3:10.5f}{4:10.5f}{5:10.5f}{6:10.5f}{7:10.5f}\n").format(scoreFinalMean, xMean[0], xMean[1], xMean[2], xMean[3], xMean[4], xMean[5], xMean[6]))
				out2.write(("{0:10.5f}{1:10.5f}{2:10.5f}{3:10.5f}{4:10.5f}{5:10.5f}{6:10.5f}{7:10.5f}\n").format(scoreFinalSTD, xSTD[0], xSTD[1], xSTD[2], xSTD[3], xSTD[4], xSTD[5], xSTD[6]))
			else:
				print('Check fitted model!')


def printFinalSimulation(DTsimOpt1, DTsimOpt2, stimulus_value_pair_1, stimulus_value_pair_2):
	with open('OutDTsimOpt1_FINAL_V1_'+str(stimulus_value_pair_1)+'_V2_'+str(stimulus_value_pair_2)+'.csv', 'w') as out3:
		for kk in range(len(DTsimOpt1)):
			out3.write(("{0:10.5f}\n").format(DTsimOpt1[kk]))
	with open('OutDTsimOpt2_FINAL_V1_'+str(stimulus_value_pair_1)+'_V2_'+str(stimulus_value_pair_2)+'.csv', 'w') as out4:
		for kk in range(len(DTsimOpt2)):
			out4.write(("{0:10.5f}\n").format(DTsimOpt2[kk]))


def initialiseModelParams(fittedModel, dataGeneratingModel):
	#initialise fixed model parameters
	pFixedDict = {}

	#initialise model paramteters appearing in all models
	gamInit = 0.5

	#model-specific parameters
	#if fittedModel == 'pDDM' or dataGeneratingModel == 'pDDM':
	#	fixedAdditiveNoise = True
	#else:
	#	fixedAdditiveNoise = False
	
	fixedAdditiveNoise = True
	if fittedModel == 'LCA':
		thrInit = 0.401
		sigInit = 0.1/np.sqrt(2)
		spvInit = 0.1
	else:
		thrInit = 0.301
		driftVarInit = 0.05
		sigInit = 0.1
		spvInit = 0.075

	if fittedModel in ['mDDM', 'mSOU', 'mUOU']:
		noiseCoeffInit = 0.125

	if fittedModel == 'pDDM':
		pVarDict = {'threshInit': thrInit, 'gammaInit': gamInit,
				'spvInit': spvInit, 'driftVarInit': driftVarInit}
	elif fittedModel == 'mDDM':
		pVarDict = {'threshInit': thrInit, 'gammaInit': gamInit,
				'spvInit': spvInit, 'driftVarInit': driftVarInit,
				'noiseCoeffInit': noiseCoeffInit}
	elif fittedModel == 'mSOU':
		thrInit = thrInit/2.0
		spvInit = spvInit/2.0
		decayInit = -2.5
		pVarDict = {'threshInit': thrInit, 'gammaInit': gamInit,
				'spvInit': spvInit, 'driftVarInit': driftVarInit,
				'noiseCoeffInit': noiseCoeffInit, 'decayInit': decayInit}
	elif fittedModel == 'mUOU':
		growthInit = 2.5
		pVarDict = {'threshInit': thrInit, 'gammaInit': gamInit,
				'spvInit': spvInit, 'driftVarInit': driftVarInit,
				'noiseCoeffInit': noiseCoeffInit, 'growthInit': growthInit}
	elif fittedModel == 'LCA':
		betaInit = 5.0
		pVarDict = {'threshInit': thrInit, 'gammaInit': gamInit,
				'spvInit': spvInit, 'betaInit': betaInit}
	else:
		print('Check available models: pDDM, mDDM, mSOU, mUOU, LCA')

	if fixedAdditiveNoise == True:
		pVarDict['sigmaInit'] = None
		pFixedDict["sigmaInit"] = sigInit
	else:
		pVarDict['sigmaInit'] = sigInit
		pFixedDict["sigmaInit"] = None

	return pVarDict, pFixedDict

