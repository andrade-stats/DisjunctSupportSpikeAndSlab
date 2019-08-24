import numpy
from SpikeAndSlabNonContinuous_ModelSearch_Proposed import SpikeAndSlabProposedModelSearch as SpikeAndSlabProposedModelSearch_NONCONT
import shared.idcHelper as idcHelper

MAX_INCREASE_IN_ERROR = 0.05

def select(maxIncreaseInError, allResultsForEachDelta, repId, variableNames = None):
    bestSelectedVars = None
    bestDelta = None
    bestCorrespondingErrorIncrease = None
    smallestIncreaseInError = numpy.inf
    smallestIncreaseInErrorModel = None
    for delta in SpikeAndSlabProposedModelSearch_NONCONT.allDelta:
        
        if repId is not None:
            selectedVars, estimatedSigmaSquareR_BMA, estimatedSigmaSquareR_reducedModel, sortedAssignmentsByFrequency = (allResultsForEachDelta[delta])[repId]
            _, sigmaSquareR_trueEst, _, _ = (allResultsForEachDelta[0.0])[repId]
        else:
            selectedVars, estimatedSigmaSquareR_BMA, estimatedSigmaSquareR_reducedModel, sortedAssignmentsByFrequency = allResultsForEachDelta[delta]
            _, sigmaSquareR_trueEst, _, _ = allResultsForEachDelta[0.0]
            
        increaseInError = (estimatedSigmaSquareR_reducedModel / sigmaSquareR_trueEst) - 1.0
        increaseInError = numpy.max([increaseInError, 0.0])
        
        print("--")
        print("delta = ", delta)
        print("selectedVars = ", selectedVars)
        if variableNames is not None:
            idcHelper.showSelectedVariables(variableNames, selectedVars)
            
        print("true MSE estimate = ", sigmaSquareR_trueEst)
        print("MSE simplified model = ", estimatedSigmaSquareR_reducedModel)
        print("increaseInError (in percent) = " + str(round(increaseInError * 100, 2)) + "\\%")
        
        if increaseInError < maxIncreaseInError:
            if (bestDelta is None) or (len(selectedVars) < len(bestSelectedVars)) or (len(selectedVars) == len(bestSelectedVars) and increaseInError < bestCorrespondingErrorIncrease):
                bestSelectedVars = selectedVars
                bestDelta = delta
                bestCorrespondingErrorIncrease = increaseInError
        
        # backup in case all models violate maxIncreaseInError
        if increaseInError < smallestIncreaseInError:
            smallestIncreaseInErrorModel = selectedVars
            smallestIncreaseInError = increaseInError
        
    assert(smallestIncreaseInErrorModel is not None)
    if bestSelectedVars is None:
        print("WARNING ALL MODELS FOUND VIOLATE MINIMUM INCREASE IN MSE REQUIREMENT")
        bestSelectedVars = smallestIncreaseInErrorModel
        
    print("===")
    print("bestSelectedVars = ", bestSelectedVars)
    if variableNames is not None:
        idcHelper.showSelectedVariables(variableNames, bestSelectedVars)
    
    if bestCorrespondingErrorIncrease is None:
        print("no bestCorrespondingErrorIncrease available")
    else:
        print("bestCorrespondingErrorIncrease = ", bestCorrespondingErrorIncrease)
        print("bestCorrespondingErrorIncrease (in percent) = " + str( round(bestCorrespondingErrorIncrease * 100.0,2) ) + "%")

    return bestSelectedVars, bestDelta, bestCorrespondingErrorIncrease