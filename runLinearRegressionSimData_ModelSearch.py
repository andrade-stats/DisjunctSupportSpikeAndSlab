import shared.idcHelper as idcHelper
import shared.regressionHelper as regressionHelper
import shared.statHelper as statHelper

import multiprocessing
import parallelEvaluation
from SpikeAndSlabNonContinuous_ModelSearch_Proposed import SpikeAndSlabProposedModelSearch as SpikeAndSlabProposedModelSearch_NONCONT
import numpy

import baselines
import simDataGeneration
import sys
import rpy2
print("rpy2 version = " + rpy2.__version__)
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

# comment out this line if the baseline methods are not required
ro.r('source(\'imports.R\')')

ro.r('set.seed(8985331)') # set a fixed random seed to make results of glasso, stars etc reproducible 


import pickle
import deltaSelection

# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.2 SSLASSO

# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.5 SSLASSO
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.5 SSLASSO

# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.0 proposed_continuous 0.5 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.2 proposed_continuous 0.5 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.5 proposed_continuous 0.5 10000

# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.0 proposed_continuous 0.5 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.2 proposed_continuous 0.5 10000

# NEW REP 10 EXPERIMENTS start July 28th !!
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.0 proposed 0.8 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.0 proposed 0.5 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.0 proposed 0.05 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.0 proposed 0.01 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.0 proposed 0.001 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.0 proposed 0.0 10000

# FINISHED:
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.2 proposed 0.8 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.2 proposed 0.5 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.2 proposed 0.05 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.2 proposed 0.01 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.2 proposed 0.001 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.2 proposed 0.0 10000



# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.0 proposed 0.8 1000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.0 proposed 0.5 1000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.0 proposed 0.05 1000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.0 proposed 0.01 1000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.0 proposed 0.001 1000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.0 proposed 0.0 1000

# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.2 proposed 0.8 1000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.2 proposed 0.5 1000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.2 proposed 0.05 1000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.2 proposed 0.01 1000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.2 proposed 0.001 1000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.2 proposed 0.0 1000


# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.0 proposed 0.0 1000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.2 proposed 0.0 1000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.5 proposed 0.0 1000

# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.5 proposed 0.0 1000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.2 proposed 0.0 1000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.0 proposed 0.0 1000

# currently running
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.0 proposed 0.8 1000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.0 proposed 0.5 1000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.0 proposed 0.05 1000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.0 proposed 0.01 1000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.0 proposed 0.001 1000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.0 proposed 0.0 1000

# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.0 horseshoe
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.2 horseshoe

# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.0 GibbsBVS
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.2 GibbsBVS
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.5 GibbsBVS

# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.0 EBIC

# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.0 EBIC
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.5 EMVS

# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.0 stabilitySelection
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.2 stabilitySelection

# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.0 EBIC
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.2 EBIC
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.5 EBIC

# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.0 stability
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.2 stability



# ********************************
# finished MCMC 10000 experiments:
# ********************************

# currently running:
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.2 proposed 0.8 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.2 proposed 0.01 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.2 proposed 0.001 10000

# on mac server:
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.2 proposed 0.8 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.2 proposed 0.01 10000

# FNISHED:
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.2 proposed 0.5 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.2 proposed 0.05 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.2 proposed 0.0 10000

# FINISHED
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.0 proposed 0.001 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.0 proposed 0.8 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.0 proposed 0.5 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.0 proposed 0.05 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.0 proposed 0.01 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.0 proposed 0.0 10000

# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.0 proposed 0.8 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.0 proposed 0.5 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.0 proposed 0.05 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.0 proposed 0.01 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.0 proposed 0.001 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.0 proposed 0.0 10000

# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.2 proposed 0.8 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.2 proposed 0.5 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.2 proposed 0.05 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.2 proposed 0.01 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.2 proposed 0.001 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.2 proposed 0.0 10000

# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.5 proposed 0.8 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.5 proposed 0.5 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.5 proposed 0.05 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.5 proposed 0.01 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.5 proposed 0.001 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.5 proposed 0.0 10000

# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.0 GibbsBvs
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.2 GibbsBvs
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.5 GibbsBvs

# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.0 EMVS
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.2 AIC
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.5 AIC

# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.2 EBIC
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.0 EBIC

# FINISHED AND UPDATED IN LATEX:
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.0 horseshoe
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.2 horseshoe
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.5 horseshoe
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.0 horseshoe
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.2 horseshoe

if len(sys.argv) > 1:
    
    dataType = sys.argv[1]
    
    noiseRatio = float(sys.argv[2])
    
    method = sys.argv[3]
    
    if method == "proposed" or method == "proposed_continuous":
        delta = float(sys.argv[4])
        assert(delta >= 0.0)
        assert(delta <= 0.8)
        
        NR_MCMC_SAMPLES = int(sys.argv[5])
        assert(NR_MCMC_SAMPLES == 10000) 

else:
    # dataType = "highDim"
    dataType = "correlated"
    noiseRatio = 0.0
    method = "SSLASSO"
    assert(False)

if dataType == "correlated" or dataType == "oneHuge":
    allN = [10, 50, 100, 1000, 100000]
elif dataType == "highDim" or dataType == "highDimOr":
    allN = [100, 1000]
else:
    assert(False)


NUMBER_OF_REPETITIONS = 10

# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.2 proposed_continuous 0.5 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.0 proposed_continuous 0.5 10000

# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.2 EBIC
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.0 EBIC


# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.0 EMVS
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.2 EMVS
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.5 EMVS


# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.0 proposed 0.5 10000
# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.5 proposed 0.5 10000

# [0.8, 0.5, 0.05, 0.01, 0.001, 0.0]
# sleep 1h; /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.2 proposed 0.8 10000
# sleep 1h; /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.2 proposed 0.05 10000
# sleep 1h; /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.2 proposed 0.01 10000
# sleep 4h; /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.2 proposed 0.001 10000
# sleep 4h; /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.2 proposed 0.0 10000

# sleep 4h; /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.5 proposed 0.8 10000
# sleep 8h; /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.5 proposed 0.05 10000
# sleep 8h; /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.5 proposed 0.01 10000
# sleep 8h; /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.5 proposed 0.001 10000
# sleep 12h; /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.5 proposed 0.0 10000

# sleep 12h; /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.0 proposed 0.8 10000
# sleep 12h; /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.0 proposed 0.05 10000
# sleep 15h; /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.0 proposed 0.01 10000
# sleep 15h; /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.0 proposed 0.001 10000
# sleep 15h; /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py correlated 0.0 proposed 0.0 10000


# /opt/intel/intelpython3/bin/python runLinearRegressionSimData_ModelSearch.py highDim 0.2 SSLASSO

from pathlib import Path
my_file = Path("saveSigma0Values")
if not my_file.is_file():
    print("saveSigma0Values file not found -> set file")
    SpikeAndSlabProposedModelSearch_NONCONT.saveSigma0Values()


if method == "proposed" or method == "proposed_continuous":
    allSettingsStr = ["delta = " + str(delta)]
elif method == "horseshoe":
    allSettingsStr = ["delta = " + str(d) for d in SpikeAndSlabProposedModelSearch_NONCONT.allDelta]
    allSettingsStr.append("automatic")
elif method == "EBIC":
    allSettingsStr = ["$\\gamma = " + str(gamma) + "$" for gamma in [0.0, 0.5, 1.0]]
elif method == "stability":
    if dataType == "highDim":
        qValues = [1,50,100]
    else:
        qValues = [1,4,6]
    allSettingsStr = ["$q = " + str(q) + "$" for q in qValues]
else:
    allSettingsStr = ["no delta"]
                
selectedVars_forAllSettings = {}
numpyAllF1Scores_forAllSettings = {}
nrVariables_forAllSettings = {}
for setting in allSettingsStr:
    numpyAllF1Scores_forAllSettings[setting] = numpy.zeros((len(allN), NUMBER_OF_REPETITIONS)) * numpy.nan
    nrVariables_forAllSettings[setting] = numpy.zeros((len(allN), NUMBER_OF_REPETITIONS)) * numpy.nan
    selectedVars_forAllSettings[setting] = {}
    
pool = multiprocessing.Pool(processes=NUMBER_OF_REPETITIONS)

for nId in range(len(allN)):
    
    n = allN[nId]
    
    trueBetaWithoutNoiseOrOutlier, allX, allY, trueBeta, correlationMatrix, responseStd = simDataGeneration.getSyntheticData(dataType, noiseRatio, n, NUMBER_OF_REPETITIONS)
    
    filenameStem = "resultsSimData/" + dataType + "_" + str(noiseRatio) + "_" + str(n) 
    
    
    p = allX[0].shape[1]
    trueRelevantVariables = numpy.where(trueBetaWithoutNoiseOrOutlier != 0)[0]
    
    # print("trueRelevantVariables = ", trueRelevantVariables)
    # assert(False)
    
    if method == "proposed":
        
        assert(NR_MCMC_SAMPLES == 10000)
        allResults = parallelEvaluation.runProposedMethod_MCMC_SEARCH(parallelEvaluation.searchBestModel_NONCONT, allY, allX, delta, NR_MCMC_SAMPLES, pool)
        
        # save all results
        pickle.dump( allResults, open( filenameStem + "_" + str(NR_MCMC_SAMPLES) + "_proposed_delta" + str(delta), "wb" ) )
        
        for repId in range(NUMBER_OF_REPETITIONS):
            selectedVars = allResults[repId][0]
            numpyAllF1Scores_forAllSettings["delta = " + str(delta)][nId, repId] = statHelper.getVariablesF1Score(p, trueRelevantVariables, selectedVars)
            nrVariables_forAllSettings["delta = " + str(delta)][nId, repId] = selectedVars.shape[0]
    
    elif method == "proposed_continuous":
        
        assert(NR_MCMC_SAMPLES == 10000)
        allResults = parallelEvaluation.runProposedMethod_MCMC_SEARCH(parallelEvaluation.searchBestModel_CONTINUOUS, allY, allX, delta, NR_MCMC_SAMPLES, pool)
        
        # save all results
        pickle.dump( allResults, open( filenameStem + "_" + str(NR_MCMC_SAMPLES) + "_proposed_continuous_delta" + str(delta), "wb" ) )
        
        for repId in range(NUMBER_OF_REPETITIONS):
            selectedVars = allResults[repId][0]
            numpyAllF1Scores_forAllSettings["delta = " + str(delta)][nId, repId] = statHelper.getVariablesF1Score(p, trueRelevantVariables, selectedVars)
            nrVariables_forAllSettings["delta = " + str(delta)][nId, repId] = selectedVars.shape[0]
                    
    elif method == "horseshoe":
        
        MCMC_SAMPLES = 10000
        
        # cannot be applied due to memory issues
        # if n == 100000:
        #    continue
        
        allResultsForEachDelta = {}
        for delta in SpikeAndSlabProposedModelSearch_NONCONT.allDelta:
            allResultsForEachDelta[delta] = [None] * NUMBER_OF_REPETITIONS
        
            
        for repId in range(NUMBER_OF_REPETITIONS):
            X = allX[repId]
            y = allY[repId]
            
            selectedVars_credibilityInterval, meanBeta, Sigma2Hat_fullModel = baselines.runHorseshoe(X, y, MCMC_SAMPLES)
            
            
            for delta in SpikeAndSlabProposedModelSearch_NONCONT.allDelta:
                selectedVars = numpy.where(numpy.abs(meanBeta) > delta)[0]
                numpyAllF1Scores_forAllSettings["delta = " + str(delta)][nId, repId] = statHelper.getVariablesF1Score(p, trueRelevantVariables, selectedVars)
                nrVariables_forAllSettings["delta = " + str(delta)][nId, repId] = selectedVars.shape[0]
                selectedVars_forAllSettings["delta = " + str(delta)][repId] = selectedVars
                
                _, _, Sigma2Hat_subModel = baselines.runHorseshoe(X[:,selectedVars], y, MCMC_SAMPLES)
                allResultsForEachDelta[delta][repId] = (selectedVars, Sigma2Hat_fullModel, Sigma2Hat_subModel, None)
        
        for repId in range(NUMBER_OF_REPETITIONS):
            selectedVars, _, _ = deltaSelection.select(deltaSelection.MAX_INCREASE_IN_ERROR, allResultsForEachDelta, repId)
            numpyAllF1Scores_forAllSettings["automatic"][nId, repId] = statHelper.getVariablesF1Score(p, trueRelevantVariables, selectedVars)
            nrVariables_forAllSettings["automatic"][nId, repId] = selectedVars.shape[0]
            selectedVars_forAllSettings["automatic"][repId] = selectedVars
        
        
        
        
    elif method == "EBIC":
        for repId in range(NUMBER_OF_REPETITIONS):
            X = allX[repId]
            y = allY[repId]
            
            allNonZeroPositions_foundByLARS = baselines.runLars(X, y)
            totalNrEstimates = len(allNonZeroPositions_foundByLARS)
            p = X.shape[1]
            
            for gamma in [0.0, 0.5, 1.0]:
                allLogMarginal_EBIC = numpy.zeros(totalNrEstimates)
                for larsEstId in range(totalNrEstimates):
                    nonZeroPositions = allNonZeroPositions_foundByLARS[larsEstId]
                    estimatedZeros = numpy.delete(numpy.arange(p), nonZeroPositions)
                    allLogMarginal_EBIC[larsEstId] = - 0.5 * regressionHelper.getLinearRegEBIC(X, y, estimatedZeros, gamma)
                selectedVars = allNonZeroPositions_foundByLARS[numpy.argmax(allLogMarginal_EBIC)]
                
                numpyAllF1Scores_forAllSettings["$\\gamma = " + str(gamma) + "$"][nId, repId] = statHelper.getVariablesF1Score(p, trueRelevantVariables, selectedVars)
                nrVariables_forAllSettings["$\\gamma = " + str(gamma) + "$"][nId, repId] = selectedVars.shape[0]
                selectedVars_forAllSettings["$\\gamma = " + str(gamma) + "$"][repId] = selectedVars
    
    elif method == "stability":
        
        for repId in range(NUMBER_OF_REPETITIONS):
        
            for q in qValues:
                selectedVars = baselines.stabilitySelection(allX[repId], allY[repId], q)
                numpyAllF1Scores_forAllSettings["$q = " + str(q) + "$"][nId, repId] = statHelper.getVariablesF1Score(p, trueRelevantVariables, selectedVars)
                nrVariables_forAllSettings["$q = " + str(q) + "$"][nId, repId] = selectedVars.shape[0]
                selectedVars_forAllSettings["$q = " + str(q) + "$"][repId] = selectedVars
                
    else:
        
        for repId in range(NUMBER_OF_REPETITIONS):
            if method == "EMVS":
                selectedVars = baselines.runEMVS(allX[repId], allY[repId])
            elif method == "SSLASSO":
                selectedVars = baselines.runSSLASSO(allX[repId], allY[repId])
            elif method == "GibbsBvs":
                
                MCMC_SAMPLES = 10000
                
                if n >= 100000:
                    selectedVars = None
                else:
                    selectedVars = baselines.runGibbsBvs(allX[repId], allY[repId], MCMC_SAMPLES)
            
            elif method == "AIC":
                X = allX[repId]
                y = allY[repId]
                
                allNonZeroPositions_foundByLARS = baselines.runLars(X, y)
                totalNrEstimates = len(allNonZeroPositions_foundByLARS)
                p = X.shape[1]
                
                allLogMarginal_AIC = numpy.zeros(totalNrEstimates)
                for larsEstId in range(totalNrEstimates):
                    nonZeroPositions = allNonZeroPositions_foundByLARS[larsEstId]
                    estimatedZeros = numpy.delete(numpy.arange(p), nonZeroPositions)
                    allLogMarginal_AIC[larsEstId] = - 0.5 * regressionHelper.getLinearRegAIC(X, y, estimatedZeros)
                selectedVars = allNonZeroPositions_foundByLARS[numpy.argmax(allLogMarginal_AIC)]
            else:
                assert(False)
            
            selectedVars_forAllSettings["no delta"][repId] = selectedVars
            
            if selectedVars is None:
                numpyAllF1Scores_forAllSettings["no delta"][nId, repId] = numpy.nan
                nrVariables_forAllSettings["no delta"][nId, repId] = numpy.nan
            else:
                numpyAllF1Scores_forAllSettings["no delta"][nId, repId] = statHelper.getVariablesF1Score(p, trueRelevantVariables, selectedVars)
                nrVariables_forAllSettings["no delta"][nId, repId] = selectedVars.shape[0]
                
    
    
    if not method.startswith("proposed"):
        # save all results
        pickle.dump( selectedVars_forAllSettings, open( filenameStem + "_" + method, "wb" ) )
    
    
   



print("FINISHED SUCCESSFULLT ALL EXPERIMENTS ")


allResultStr_f1 = {}
allResultStr_nrVars = {}
for setting in allSettingsStr:
    allResultStr_f1[setting] = ""
    allResultStr_nrVars[setting] = ""
    for nId in range(len(allN)):
        allResultStr_f1[setting] += " & " + idcHelper.getAvgAndStdWithDigitRound(numpyAllF1Scores_forAllSettings[setting][nId], 2)
        allResultStr_nrVars[setting] += " & " + idcHelper.getAvgAndStdWithDigitRound(nrVariables_forAllSettings[setting][nId], 2)
       


print("------------ results for latex table ------------- ")
print("")
print("F1-SCORES")
print("\\midrule")
print(" & " + " & ".join([str(n) for n in allN]) + " \\\\")
print("\\midrule")
for setting in allSettingsStr:
    if setting == "no delta":
        fullMethodStr = method
    else:
        fullMethodStr = method + " (" + setting + ") "
    print(fullMethodStr + allResultStr_f1[setting] + " \\\\")

print("")    
print("nr variables")
print("\\midrule")
print(" & " + " & ".join([str(n) for n in allN]) + " \\\\")
print("\\midrule")
for setting in allSettingsStr:
    if setting == "no delta":
        fullMethodStr = method
    else:
        fullMethodStr = method + " (" + setting + ") "
    print(fullMethodStr + allResultStr_nrVars[setting] + " \\\\")
    
print("************* SUMMARY OF SETTINGS: ****************")
print("dataType = ", dataType)
print("noiseRatio = ", noiseRatio)
print("allN = ", allN)
print("NUMBER_OF_REPETITIONS = ", NUMBER_OF_REPETITIONS)
print("method = ", method)
if method.startswith("proposed"):
    print("delta = ", delta)
    print("NR_MCMC_SAMPLES = ", NR_MCMC_SAMPLES)

print("FINISHED ALL EXPERIMENTS")




