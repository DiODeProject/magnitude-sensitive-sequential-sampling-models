import timeit
import sys
sys.path.append('../../LibraryModules/')
from LibModelMimicry2019 import *

rand.seed(22865198)

#specify models for data generation and fitting
dataGeneratingModel = 'mDDM'
fittedModel = "LCA"

#simulated conditions
stimulus_value_pairs = ((0.4, 0.3), (0.6, 0.5), (0.6, 0.45), (0.45, 0.45))
cond_list = ('baseline', 'additive', 'multiplicative', 'equal')

#access specific data set
chooseDataSetType = 'DDMOU' # 'DDMOU' or 'LCA'
if len(sys.argv) > 1:
	dataSetID = int(sys.argv[1])
else:
	dataSetID = 1 # this number relates to the choice of the random seed for the generated data
dataFilePath = '../../../../ModelMimicry2019_DataGeneration/' + str(chooseDataSetType) + '_DataGeneration_' + str(dataSetID) + '/' + str(chooseDataSetType)+ '-24Jun2019-' + str(dataSetID) + '-runid-'

#tolerance fitting Downhill Simplex
if len(sys.argv) > 2:
	FitTolerance = float(sys.argv[2])
else:
	FitTolerance = 200.0

#absolute tolerance for parameters in Downhill Simplex
if len(sys.argv) > 3:
	paramTol = float(sys.argv[3])
else:
	paramTol = 0.001

#upper limit on simulation time (both data and model)
if len(sys.argv) > 4:
	TMAXdec = float(sys.argv[4])
else:
	TMAXdec = 6.0

#number of trials in model fitting/obtaining initial parameter configuration
NtrialsGetInitParams = 1000 #500 

#initialise fixed model parameters
pVarDict, pFixedDict = initialiseModelParams(fittedModel, dataGeneratingModel)

runIDgroup = ([kk+1 for kk in range(len(cond_list))], [kk+5 for kk in range(len(cond_list))], [kk+9 for kk in range(len(cond_list))], [kk+13 for kk in range(len(cond_list))], [kk+17 for kk in range(len(cond_list))])
chooseRunIDgroup = 2 # 0: sOU data, 1: uOU data, 2: mDDM data, 3: pDDM data, 4: LCA data 

time_GetInit_start = timeit.default_timer()

bin_edges_allConds_Opt1, abs_freq_allConds_Opt1, bin_edges_allConds_Opt2, abs_freq_allConds_Opt2 = \
getDataStats(chooseDataSetType, dataFilePath, runIDgroup, chooseRunIDgroup, stimulus_value_pairs, TMAXdec)

printPart1A()

parameterInitLists, param_names, fInitList = func_find_init_params(pVarDict, pFixedDict, fittedModel, NtrialsGetInitParams,
                                                                   TMAXdec,stimulus_value_pairs, cond_list,
                                                                   bin_edges_allConds_Opt1, abs_freq_allConds_Opt1,
                                                                   bin_edges_allConds_Opt2, abs_freq_allConds_Opt2,
                                                                   scale=0.95, keepInd=15, iterations=40, nrChildren=3)  #scale=0.95, keepInd=15, iterations=40, nrChildren=3
xStart0 = parameterInitLists[0]
xStart1 = parameterInitLists[1]
xStart2 = parameterInitLists[2]

# more accurate computation for best 3 parameter sets with larger number of trials
NtrialsGetInitBest3 = 20000 #5000

fVal0 = Fobj_AIC_BIC(xStart0, param_names, pFixedDict, fittedModel, NtrialsGetInitBest3, TMAXdec, stimulus_value_pairs, cond_list,
                    bin_edges_allConds_Opt1, abs_freq_allConds_Opt1, bin_edges_allConds_Opt2, abs_freq_allConds_Opt2)
fVal1 = Fobj_AIC_BIC(xStart1, param_names, pFixedDict, fittedModel, NtrialsGetInitBest3, TMAXdec, stimulus_value_pairs, cond_list,
                    bin_edges_allConds_Opt1, abs_freq_allConds_Opt1, bin_edges_allConds_Opt2, abs_freq_allConds_Opt2)
fVal2 = Fobj_AIC_BIC(xStart2, param_names, pFixedDict, fittedModel, NtrialsGetInitBest3, TMAXdec, stimulus_value_pairs, cond_list,
                    bin_edges_allConds_Opt1, abs_freq_allConds_Opt1, bin_edges_allConds_Opt2, abs_freq_allConds_Opt2)
fValList = [fVal0, fVal1, fVal2]
xStartList = [xStart0, xStart1, xStart2]
L_minStart = np.min(fValList)
xStartIndex = np.argmin(fValList)
xStart = xStartList[xStartIndex]

parameterListString = str(param_names)

printPart1B(L_minStart, xStart, parameterListString, param_names, fittedModel)
            
time_GetInit_stop = timeit.default_timer()

printPart1C(time_GetInit_stop, time_GetInit_start, parameterListString, NtrialsGetInitBest3, xStart, L_minStart, NtrialsGetInitParams, fInitList, fValList, xStartList)

printPart2A()

time_Fit_start = timeit.default_timer()

REPETITIONS = 6
SCORE_FINAL_LIST = []
X_FINAL_LIST = []

# Ntrials for Downhill-Simplex Fit
NtrialsSimplex = 10000 #20000

SCORE_start = Fobj_AIC_BIC(xStart, param_names, pFixedDict, fittedModel, NtrialsSimplex, TMAXdec, stimulus_value_pairs, cond_list,
                          bin_edges_allConds_Opt1, abs_freq_allConds_Opt1, bin_edges_allConds_Opt2, abs_freq_allConds_Opt2)

for rep in range(REPETITIONS):
	FitRes = minimize(Fobj_AIC_BIC, xStart, args=(param_names, pFixedDict, fittedModel, NtrialsSimplex, TMAXdec,
		stimulus_value_pairs, cond_list,
		bin_edges_allConds_Opt1, abs_freq_allConds_Opt1,
		bin_edges_allConds_Opt2, abs_freq_allConds_Opt2),
		method='Nelder-Mead', options={'maxiter': 10000, 'maxfev': 20000, 'ftol': FitTolerance, 'xtol': paramTol}) # 'ftol': 100, 'xtol': 1e-3

	SCORE_FINAL_LIST.append(FitRes.fun)
	xFinal = FitRes.x
	xFinal = limitToPosNegValues(param_names, xFinal)
	X_FINAL_LIST.append(xFinal)
	
	printPart2B(rep, NtrialsSimplex, parameterListString, xStart, xFinal, FitRes.fun, FitRes.message, FitRes.status, FitRes.nit, FitRes.nfev)

time_Fit_stop = timeit.default_timer()
printPart2C(time_Fit_stop, time_Fit_start)

scoreFinalMean = np.mean(SCORE_FINAL_LIST)
scoreFinalSTD = np.std(SCORE_FINAL_LIST)
xMean = np.mean(X_FINAL_LIST, axis=0)
xSTD = np.std(X_FINAL_LIST, axis=0)
printPart3A(scoreFinalMean, scoreFinalSTD, xMean, xSTD, param_names, fittedModel, parameterListString)

xInput = list(xMean)
fitParams = {}
for name, val in zip(param_names, xInput):
	fitParams[str(name)] = val

NtrialsFinalSimMeanParams = 20000 #20000
for jj in range(len(stimulus_value_pairs)):
	DTsimOpt1, DTsimOpt2 = NeuroModel(NtrialsFinalSimMeanParams, stimulus_value_pairs[jj],
			fitParams, pFixedDict, fittedModel, TMAXdec)
	printFinalSimulation(DTsimOpt1, DTsimOpt2, stimulus_value_pairs[jj][0], stimulus_value_pairs[jj][1])
