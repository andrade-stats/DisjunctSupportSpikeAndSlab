import numpy
# import autograd.numpy as numpy  # Thinly-wrapped numpy
# from autograd import value_and_grad    # The only autograd function you may ever need
# from autograd import grad
#from autograd.differential_operators import jacobian

import shared.idcHelper as idcHelper
import scipy.stats
import scipy.special
import shared.statHelper as statHelper
import scipy.optimize
import shared.analyzeHelper as analyzeHelper
import time
from collections import defaultdict

import rpy2
print("rpy2 version = " + rpy2.__version__)
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
ro.r('source(\'imports.R\')')
ro.r('set.seed(8985331)') # set a fixed random seed to make results of glasso, stars etc reproducible 

import showResultsText

# from SpikeAndSlabNonContinuous_MCMC import SpikeAndSlabProposed as SpikeAndSlabProposed_nonContinuousMCMC
import samplingHelper


class SpikeAndSlabProposedModelSearch:
    
    ETA_SQUARE_R = 1.0
    NU_R = 1.0
    
    
    ETA_SQUARE_1 = 100.0
    NU_1 = 1.0
    
    
    def __init__(self, allObservedResponses, allObservedCovariates, delta):
        
        self.delta = delta
        
        self.nu1 = SpikeAndSlabProposedModelSearch.NU_1
        self.etaSquare1 = SpikeAndSlabProposedModelSearch.ETA_SQUARE_1
        
        assert(delta > 0.0)
        self.sigmaSquare0 = SpikeAndSlabProposedModelSearch.getSpikePriorChauchyVsNormal(delta)
        
        self.fullp = allObservedCovariates.shape[1]
        self.y = allObservedResponses
        self.n = allObservedResponses.shape[0]
        
        self.setX(allObservedCovariates)
        
        return
    
    
    
    @staticmethod
    def deltaVarianceFunc(varNotRelevant, varRelevant, delta):
        
        leftSide = scipy.stats.t.logpdf(delta, df=1.0, loc=0, scale=numpy.sqrt(varRelevant)) 
        rightSide = scipy.stats.norm.logpdf(delta, loc=0, scale=numpy.sqrt(varNotRelevant)) 
        
        return leftSide - rightSide
    
    # test for delta in {0.1, 0.05, 0.01}
    @staticmethod
    def getSpikePriorChauchyVsNormal(delta):
        lowerBound = 0.000000001 
        upperBound = 0.99999999 * SpikeAndSlabProposedModelSearch.ETA_SQUARE_1
        
        # funcAtLowerBound = SpikeAndSlabProposed.deltaVarianceFuncNormal_Truncated(lowerBound, slabPriorVariance, delta)
        # funcAtUpperBound = SpikeAndSlabProposed.deltaVarianceFuncNormal_Truncated(upperBound, slabPriorVariance, delta)
        
        assert(lowerBound < upperBound)
        # assert(funcAtLowerBound < 0) # since we expect that for the lowerBound value v0_min we have that N(delta | 0, v1) > N(delta | 0, v0_min)
        # assert(funcAtUpperBound > 0) # since we expect that for the upperBound value v0_max we have that N(delta | 0, v1) < N(delta | 0, v0_max)
        spikePriorVariance = scipy.optimize.brentq(SpikeAndSlabProposedModelSearch.deltaVarianceFunc, lowerBound, upperBound, args = (SpikeAndSlabProposedModelSearch.ETA_SQUARE_1, delta))
        
#         print("at delta:")
#         print("true density spike = ", SpikeAndSlabProposed.truncatedNormal_not_relevant_logDensity(delta, spikePriorVariance, delta))
#         print("true density slab = ", SpikeAndSlabProposed.truncatedNormal_relevant_logDensity(delta, slabPriorVariance, delta))
#         print("at 0:")
#         print("true density spike = ", SpikeAndSlabProposed.truncatedNormal_not_relevant_logDensity(delta, spikePriorVariance, 0.0))
#         print("true density slab = ", SpikeAndSlabProposed.truncatedNormal_relevant_logDensity(delta, slabPriorVariance, 0.0))
        
        print("spikePriorVariance = ", spikePriorVariance)
        print("slabPriorVariance = ", SpikeAndSlabProposedModelSearch.ETA_SQUARE_1)
        
        return spikePriorVariance
    
        
    
    def setX(self, allObservedCovariates):
        self.p = allObservedCovariates.shape[1]
        assert(allObservedCovariates.shape[0] == self.n)
        
        self.X = allObservedCovariates
        
        self.XTX = numpy.matmul(self.X.transpose(), self.X)
        self.yX = numpy.matmul(self.y.transpose(), self.X)
        self.invXTX_regularized = numpy.linalg.inv(self.XTX + numpy.eye(self.p))
        return
    
    
    
    
    
    @staticmethod
    def getSigmaSquareR_reducedModel(allObservedResponses, allObservedCovariates, delta, selectedVars, NUMBER_OF_MCMC_SAMPLES_TOTAL):
        allObservedCovariates_sub = allObservedCovariates[:, selectedVars]
        subModel = SpikeAndSlabProposedModelSearch(allObservedResponses, allObservedCovariates_sub, delta)
        return subModel.getSigmaSquareR_fullModel_fromCurrentModel(NUMBER_OF_MCMC_SAMPLES_TOTAL)
    
    
    # UDATED FOR CONTINUOUS
    def sampleZ(self, NUMBER_OF_MCMC_SAMPLES_TOTAL):
        
        invEst = numpy.linalg.inv(self.X.transpose() @ self.X + 1.0 * numpy.eye(self.p))
        ridgeBetaEst = (invEst @ self.X.transpose()) @ self.y
        
        z = numpy.zeros(self.p, dtype = numpy.int)
        z[numpy.absolute(ridgeBetaEst) > self.delta] = 1
        
        beta = ridgeBetaEst
        
        # get a sparse initial solution in order to ensure faster convergence
        maxNrInitialSelectedVars = int(self.p * 0.01)
        if maxNrInitialSelectedVars > 0 and numpy.sum(z) > maxNrInitialSelectedVars:
            largestIds = numpy.argsort(-numpy.absolute(ridgeBetaEst))[0:maxNrInitialSelectedVars]
            z = numpy.zeros(self.p, dtype = numpy.int)
            z[largestIds] = 1
            
        beta[z == 0] = 0
            
        sigmaSquareR = numpy.mean(numpy.square(self.y - self.X @ beta))
        
        print("beta = ")
        print(beta)
        
        print("sigmaSquareR = ")
        print(sigmaSquareR)
        
        print("z = ")
        print(z)
        
        BURN_IN_SAMPLES = int(0.1 * NUMBER_OF_MCMC_SAMPLES_TOTAL)
        assert(BURN_IN_SAMPLES >= 1)
        NUMBER_OF_MCMC_SAMPLES_USED = NUMBER_OF_MCMC_SAMPLES_TOTAL - BURN_IN_SAMPLES
        
        print("BURN_IN_SAMPLES = ", BURN_IN_SAMPLES)
        print("NUMBER_OF_MCMC_SAMPLES_USED = ", NUMBER_OF_MCMC_SAMPLES_USED)
        
        posteriorAssignments = numpy.zeros((NUMBER_OF_MCMC_SAMPLES_USED, self.p))
        averagePosteriorBeta = numpy.zeros(self.p)
        averageSigmaSquareR = 0.0
        
        spikeAndSlabVar = numpy.asarray([self.sigmaSquare0, self.etaSquare1])
        
        print("spikeAndSlabVar = ", spikeAndSlabVar)
        
        for mcmcIt in range(NUMBER_OF_MCMC_SAMPLES_TOTAL):
            print("mcmcIt = ", mcmcIt)
            
            # if self.delta > 0:
            for j in range(self.p):
                # sample p(z_j | beta, z_-j, y, sigmaSquareR, X)
                z[j] = self.sampleZjConditionedOnRest(sigmaSquareR, spikeAndSlabVar, beta, z, j)
                
                # sample p(beta_j | beta_-j, z, y, sigmaSquareR, X)
                meanTilde, sigmaSquareTilde, _ =  self.getMeanAndVarOfBetaConditional(sigmaSquareR, spikeAndSlabVar, beta, z, j)
                beta[j] = scipy.stats.norm.rvs(loc=meanTilde, scale=numpy.sqrt(sigmaSquareTilde)) 
            

            if self.delta == 0:
                # safety check for delta == 0
                assert(numpy.all(beta[z == 0] == 0) and numpy.all(beta[z == 1] != 0))
                
            
            # sample p(sigmaSquareR | beta, z, y, X)
            etaSquareForsigmaSquareR = (SpikeAndSlabProposedModelSearch.NU_R * SpikeAndSlabProposedModelSearch.ETA_SQUARE_R + numpy.sum(numpy.square(self.y - numpy.matmul(self.X, beta)))) / (SpikeAndSlabProposedModelSearch.NU_R + self.n)
            sigmaSquareR = samplingHelper.getScaledInvChiSquareSample(nu = SpikeAndSlabProposedModelSearch.NU_R + self.n, etaSquare = etaSquareForsigmaSquareR, numberOfSamples = 1)[0]
            
            # sample p(sigmaSquare_0 | beta, z, y, X) and p(sigmaSquare_1 | beta, z, y, X)
            spikeAndSlabVar[1] = self.sampleSigmaSquareConditional(beta, z)
            
            print("slab variance = ", spikeAndSlabVar[1])
            
            if mcmcIt >= BURN_IN_SAMPLES:
                posteriorAssignments[mcmcIt - BURN_IN_SAMPLES] = z
                averagePosteriorBeta += beta
                averageSigmaSquareR += sigmaSquareR
        
        averagePosteriorBeta = averagePosteriorBeta / float(NUMBER_OF_MCMC_SAMPLES_USED)
        averageSigmaSquareR = averageSigmaSquareR / float(NUMBER_OF_MCMC_SAMPLES_USED)
        
        # print("posteriorAssignments = ")
        # print(posteriorAssignments)
        
        # print("averagePosteriorBeta = ")
        # print(averagePosteriorBeta)
        
        countAssignments = defaultdict(lambda: 0)
        for mcmcIt in range(NUMBER_OF_MCMC_SAMPLES_USED):
            nonZeroPos = numpy.where(posteriorAssignments[mcmcIt] != 0)[0]
            nonZeroPosAsStr = [str(num) for num in nonZeroPos]
            nonZeroPosAsStr = " ".join(nonZeroPosAsStr)
            countAssignments[nonZeroPosAsStr] += 1
        
        sortedAssignmentsByFrequency = sorted(countAssignments.items(), key=lambda kv: kv[1], reverse = True)
        print("sortedAssignmentsByFrequency = ")
        print(sortedAssignmentsByFrequency)
        
        mostFrequentAssignment = showResultsText.getNumpyArray(sortedAssignmentsByFrequency[0][0])
        # print("mostFrequentAssignment = ", mostFrequentAssignment)
        
        # see "Optimal predictive model selection", 2004
        assignmentProbs = numpy.mean(posteriorAssignments, axis = 0)
        medianProbabilityModel = numpy.where(assignmentProbs > 0.5)[0]
        # print("assignmentProbs = ", assignmentProbs)
        # print("medianProbabilityModel = ", medianProbabilityModel)
             
        return mostFrequentAssignment, medianProbabilityModel, assignmentProbs, averagePosteriorBeta, averageSigmaSquareR, sortedAssignmentsByFrequency
    
    
    
    # UPDATED FOR CONTINUOUS
    # BRAND-NEW CHECKED
    # get mean and variance of p(beta_j | beta_-j, z, y, sigmaSquareR, X)
    def getMeanAndVarOfBetaConditional(self, sigmaSquareR, spikeAndSlabVar, beta, z, j):
        if self.delta == 0 and z[j] == 0:
            assert(spikeAndSlabVar[z[j]] is None)
            return None, None, None
        
        minusJ = numpy.delete(numpy.arange(self.p), j)
        betaMinusJ = beta[minusJ]
        XminusJ = self.X[:, minusJ]
        yTilde = self.y - numpy.matmul(XminusJ, betaMinusJ)
        xJ = self.X[:,j]
        
        yTildeTimesXj = numpy.dot(yTilde, xJ)
         
        sigmaSquareTilde = sigmaSquareR / (numpy.sum(numpy.square(xJ)) + (sigmaSquareR / spikeAndSlabVar[z[j]]))
        meanTilde = (sigmaSquareTilde / sigmaSquareR) * yTildeTimesXj
        
        additionalStatisticForCondZ = (meanTilde / (2.0 * sigmaSquareR))  * yTildeTimesXj 
        return meanTilde, sigmaSquareTilde, additionalStatisticForCondZ
    
    
    # UPDATED FOR CONTINUOUS
    # BRAND-NEW CHECKED
    # sample p(z_j | beta, z_-j, y, sigmaSquareR, X)
    def sampleZjConditionedOnRest(self, sigmaSquareR, spikeAndSlabVar, beta, originalZ, j):
        
        unnormalizedLogProbZ = numpy.zeros(2)
        
        for sspInd in [0,1]:
            
            z = numpy.copy(originalZ)
            z[j] = sspInd
            
            if self.delta > 0.0 or sspInd == 1:
                meanTilde, sigmaSquareTilde, additionalStatisticForCondZ = self.getMeanAndVarOfBetaConditional(sigmaSquareR, spikeAndSlabVar, beta, z, j)
                
                unnormalizedLogProbZ[sspInd] += additionalStatisticForCondZ
                
                
                unnormalizedLogProbZ[sspInd] += 0.5 * numpy.log(2.0 * numpy.pi * sigmaSquareTilde) # SpikeAndSlabProposedModelSearch.getTruncatedNormalLogConstant(sspInd, self.delta, sigmaSquareTilde, meanTilde)
                unnormalizedLogProbZ[sspInd] -= 0.5 * numpy.log(2.0 * numpy.pi * spikeAndSlabVar[sspInd]) # SpikeAndSlabProposedModelSearch.getTruncatedNormalLogConstant(sspInd, self.delta, spikeAndSlabVar[sspInd], 0.0)
            else:
                assert(sspInd == 0 and self.delta == 0.0)
                # nothing to do
                
            # add p(z) 
            unnormalizedLogProbZ[sspInd] += SpikeAndSlabProposedModelSearch.getLogPriorZ(numpy.sum(z), self.fullp)
            
            
        if numpy.all(unnormalizedLogProbZ == float("-inf")):
            print(unnormalizedLogProbZ)
            assert(False)
        
        logNormalization = scipy.special.logsumexp(unnormalizedLogProbZ)
        zProbs = numpy.exp(unnormalizedLogProbZ - logNormalization)
        newZj = numpy.random.choice(numpy.arange(2), p=zProbs)
        
        # print("unnormalizedLogProbZ = ")
        # print(unnormalizedLogProbZ)
        # print("zProbs = ", zProbs)
        # print("numpy.arange(2) = ", numpy.arange(2))
        # print("newZj = ", newZj)
        return newZj
    
    
    
    
    
    
    
    # UPDATED FOR CONTINUOUS
    # REVISED
    # reading checked + experiment check
    # samples from p(sigma_j^2 | beta_j, y, X, S)
    # a slice sampler as in Bayesian Methods for Data Analysis, Carlin et al Third Edition, page 139
    def sampleSigmaSquareConditional(self, beta, z):
        assert(self.delta >= 0.0)
        usedBetaCount = numpy.sum(z)
        betaSquareSum = numpy.sum(numpy.square(beta[z == 1]))
        
        etaSquarePrior = self.etaSquare1
        priorNu = self.nu1
        nu = priorNu + usedBetaCount
        
        etaSquare = (priorNu * etaSquarePrior + betaSquareSum) / nu
        return samplingHelper.getScaledInvChiSquareSample(nu, etaSquare, 1)[0]
         
    
        
    
   
    





    # *****************************************************************
    # ********** METHODS FOR MARGINAL LIKELIHOOD ESTIMATION ***********
    # *****************************************************************

    
    # UPDATED FOR CONTINUOUS
    # checked
    def estimateErrorInMSE(self, selectedVars, NUMBER_OF_MCMC_SAMPLES_TOTAL):
        
        z = numpy.zeros(self.p, dtype = numpy.int)
        z[selectedVars] = 1
            
        numberOfFreeBeta = self.p
        fixedBetaPart = numpy.zeros(self.p - numberOfFreeBeta)
        fixedSigmaSquareR = None
        fixedSlabVar = None
            
        posteriorBeta, posteriorSigmaSquareR, posteriorSlabVar = self.posteriorParameterSamples(z, NUMBER_OF_MCMC_SAMPLES_TOTAL, fixedSlabVar, fixedSigmaSquareR, numberOfFreeBeta, fixedBetaPart)
        
        irrelevantPositions = numpy.delete(numpy.arange(self.p), selectedVars)
        
        irrelevantBetaSamples = posteriorBeta[:,irrelevantPositions]
        irrelevantX = self.X[:,irrelevantPositions]
        
        sampleCovX_irrelevant = numpy.cov(irrelevantX.transpose(), bias=True)
        sampleCovBeta_irrelevant = numpy.cov(irrelevantBetaSamples.transpose(), bias=True)
        
        sampleCovX_irrelevant = numpy.atleast_2d(sampleCovX_irrelevant)
        sampleCovBeta_irrelevant = numpy.atleast_2d(sampleCovBeta_irrelevant)
        estimatedAdditionalMSE = numpy.trace(numpy.matmul(sampleCovBeta_irrelevant, sampleCovX_irrelevant))
        
        estimatedSigmaSquareR = numpy.mean(posteriorSigmaSquareR)
        
        return estimatedSigmaSquareR, estimatedAdditionalMSE
    
    
    # UPDATED FOR CONTINUOUS
    def getSigmaSquareR_fullModel_fromCurrentModel(self, NUMBER_OF_MCMC_SAMPLES_TOTAL):
        z = numpy.ones(self.p, dtype = numpy.int)
        
        numberOfFreeBeta = self.p
        fixedBetaPart = numpy.zeros(self.p - numberOfFreeBeta)
        fixedSigmaSquareR = None
        fixedSlabVar = None
        posteriorBeta, posteriorSigmaSquareR, posteriorSlabVar = self.posteriorParameterSamples(z, NUMBER_OF_MCMC_SAMPLES_TOTAL, fixedSlabVar, fixedSigmaSquareR, numberOfFreeBeta, fixedBetaPart)
        
        return numpy.mean(posteriorSigmaSquareR)
        
    
    # UPDATED FOR CONTINUOUS
    # z is always considered fixed
    def posteriorParameterSamples(self, z, NUMBER_OF_MCMC_SAMPLES_TOTAL, fixedSlabVar, fixedSigmaSquareR, numberOfFreeBeta, fixedBetaPart):
        assert(fixedSlabVar is None or fixedSlabVar > 0.0)
        assert(fixedSigmaSquareR is None or fixedSigmaSquareR > 0.0)
        assert(numberOfFreeBeta + fixedBetaPart.shape[0] == self.p)
        
        invEst = numpy.linalg.inv(self.X.transpose() @ self.X + 1.0 * numpy.eye(self.p))
        ridgeBetaEst = (invEst @ self.X.transpose()) @ self.y
        
        beta = ridgeBetaEst
        beta[numberOfFreeBeta:self.p] = fixedBetaPart
        
        if fixedSigmaSquareR is None:
            sigmaSquareR = numpy.mean(numpy.square(self.y - self.X @ ridgeBetaEst))
        else:
            sigmaSquareR = fixedSigmaSquareR
        
        
        # print("z = ")
        # print(z)
        # print("beta = ")
        # print(beta)
        # assert(False)
        
        # print("sigmaSquareR = ")
        # print(sigmaSquareR)
        
        
        BURN_IN_SAMPLES = int(0.1 * NUMBER_OF_MCMC_SAMPLES_TOTAL)
        assert(BURN_IN_SAMPLES >= 1)
        NUMBER_OF_MCMC_SAMPLES_USED = NUMBER_OF_MCMC_SAMPLES_TOTAL - BURN_IN_SAMPLES
        
        # print("BURN_IN_SAMPLES = ", BURN_IN_SAMPLES)
        # print("NUMBER_OF_MCMC_SAMPLES_USED = ", NUMBER_OF_MCMC_SAMPLES_USED)
        
        posteriorBeta = numpy.zeros((NUMBER_OF_MCMC_SAMPLES_USED, self.p))
        posteriorSigmaSquareR = numpy.zeros(NUMBER_OF_MCMC_SAMPLES_USED)
        posteriorSlabVar = numpy.zeros(NUMBER_OF_MCMC_SAMPLES_USED)
        
        spikeAndSlabVar = numpy.asarray([self.sigmaSquare0, self.etaSquare1])
        if fixedSlabVar is not None:
            spikeAndSlabVar[1] = fixedSlabVar
            
        
        
        for mcmcIt in range(NUMBER_OF_MCMC_SAMPLES_TOTAL):
            print("mcmcIt = ", mcmcIt)
            
            for j in range(numberOfFreeBeta):
                
                # sample p(beta_j | beta_-j, z, y, sigmaSquareR, X)
                meanTilde, sigmaSquareTilde, _ =  self.getMeanAndVarOfBetaConditional(sigmaSquareR, spikeAndSlabVar, beta, z, j)
                beta[j] = scipy.stats.norm.rvs(loc=meanTilde, scale=numpy.sqrt(sigmaSquareTilde)) 
            
            if fixedSigmaSquareR is None:
                # sample p(sigmaSquareR | beta, z, y, X)
                etaSquareForsigmaSquareR = (SpikeAndSlabProposedModelSearch.NU_R * SpikeAndSlabProposedModelSearch.ETA_SQUARE_R + numpy.sum(numpy.square(self.y - numpy.matmul(self.X, beta)))) / (SpikeAndSlabProposedModelSearch.NU_R + self.n)
                sigmaSquareR = samplingHelper.getScaledInvChiSquareSample(nu = SpikeAndSlabProposedModelSearch.NU_R + self.n, etaSquare = etaSquareForsigmaSquareR, numberOfSamples = 1)[0]
                
            if fixedSlabVar is None:
                # sample p(sigmaSquare_1 | beta, z, y, X)
                spikeAndSlabVar[1] = self.sampleSigmaSquareConditional(beta, z)
            
            
            if mcmcIt >= BURN_IN_SAMPLES:
                posteriorBeta[mcmcIt - BURN_IN_SAMPLES] = beta
                posteriorSigmaSquareR[mcmcIt - BURN_IN_SAMPLES] = sigmaSquareR
                posteriorSlabVar[mcmcIt - BURN_IN_SAMPLES] = spikeAndSlabVar[1]
        
        return posteriorBeta, posteriorSigmaSquareR, posteriorSlabVar
    
    
    
    
    @staticmethod
    def getLogPriorZ(s, p):
        a = 1.0
        b = 1.0
        priorLogProb = scipy.special.betaln(a + s, b + p - s) - scipy.special.betaln(a, b)
        return priorLogProb
        
    
    
    
    


# DELTA_SPEC = 0.8
# SpikeAndSlabProposedModelSearch.getSpikePriorChauchyVsNormal(delta = DELTA_SPEC)
# print("CHECKED:")
# SpikeAndSlabProposedModelSearch.getSigmaSquare0(delta = DELTA_SPEC, NUMBER_OF_SAMPLES = 10000)

# rpy2 version = 2.9.4
# logProbDelta_relevant =  -3.462343595105735
# logProbAtDelta_notRelevant =  -0.9239385332046727
# FINAL ESTIMATE:
# logProbDelta_relevant =  -3.462343595105735
# logProbAtDelta_notRelevant =  -3.4616479446294974
# sigmaSquare0 =  0.000820256218981533

# SpikeAndSlabProposedModelSearch.marginalTest()
# [0,1, 2] logMarginal =  -36.493045176888565
# [0,1]  logMarginal =  -33.972331077557584
# [0] logMarginal =  -30.520632101361606
# [] logMarginal =  -33.33177281540057
