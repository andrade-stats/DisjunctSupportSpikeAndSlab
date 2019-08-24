import numpy
import multiprocessing

# proposed non-continous model
# from SpikeAndSlabNonContinuous_MCMC import SpikeAndSlabProposed as SpikeAndSlabProposed_MCMC
# from SpikeAndSlabNonContinuous_Laplace import SpikeAndSlabProposed as SpikeAndSlabProposed_Laplace
from SpikeAndSlabNonContinuous_ModelSearch_Proposed import SpikeAndSlabProposedModelSearch as SpikeAndSlabProposedModelSearch_NONCONT
from SpikeAndSlabContinuous_ModelSearch_Proposed import SpikeAndSlabProposedModelSearch as SpikeAndSlabProposedModelSearch_CONT








def searchBestModel_CONTINUOUS(allPassedData):
    
    try:
            
        y, X, delta, NR_MCMC_SAMPLES = allPassedData
        
        myModelSearch = SpikeAndSlabProposedModelSearch_CONT(y, X, delta)
        selectedVariablesProposedMethod, medianProbabilityModel, assignmentProbs, averagePosteriorBeta, estimatedSigmaSquareR_BMA, sortedAssignmentsByFrequency = myModelSearch.sampleZ(NR_MCMC_SAMPLES)
        
        estimatedSigmaSquareR_reducedModel = SpikeAndSlabProposedModelSearch_CONT.getSigmaSquareR_reducedModel(y, X, delta, selectedVariablesProposedMethod, NR_MCMC_SAMPLES)
        
    except (KeyboardInterrupt):
        print("! GOT KEYBOARD INTERRUPT OR AN EXCEPTION !")
        
    return selectedVariablesProposedMethod, estimatedSigmaSquareR_BMA, estimatedSigmaSquareR_reducedModel, sortedAssignmentsByFrequency

def searchBestModel_NONCONT(allPassedData):
    
    try:
            
        y, X, delta, NR_MCMC_SAMPLES = allPassedData
        
        myModelSearch = SpikeAndSlabProposedModelSearch_NONCONT(y, X, delta)
        selectedVariablesProposedMethod, medianProbabilityModel, assignmentProbs, averagePosteriorBeta, estimatedSigmaSquareR_BMA, sortedAssignmentsByFrequency = myModelSearch.sampleZ(NR_MCMC_SAMPLES)
        
        estimatedSigmaSquareR_reducedModel = SpikeAndSlabProposedModelSearch_NONCONT.getSigmaSquareR_reducedModel(y, X, delta, selectedVariablesProposedMethod, NR_MCMC_SAMPLES)
        
    except (KeyboardInterrupt):
        print("! GOT KEYBOARD INTERRUPT OR AN EXCEPTION !")
        
    return selectedVariablesProposedMethod, estimatedSigmaSquareR_BMA, estimatedSigmaSquareR_reducedModel, sortedAssignmentsByFrequency


def runProposedMethod_MCMC_SEARCH(searchBestModel, allY, allX, delta, NR_MCMC_SAMPLES, pool):
    
    dataForProcessingEachVariableSelection = []
    
    NR_REPETIONS = len(allY)
    
    for repId in range(NR_REPETIONS):
        y = allY[repId]
        X = allX[repId]
        allPassedData = (y, X, delta, NR_MCMC_SAMPLES)
        dataForProcessingEachVariableSelection.append(allPassedData)
    
    try:
        allResults = pool.map(searchBestModel, dataForProcessingEachVariableSelection)
    except (KeyboardInterrupt):
        print("main process exiting..")
        pool.terminate()
        pool.join()
        assert(False)
        
    return allResults
