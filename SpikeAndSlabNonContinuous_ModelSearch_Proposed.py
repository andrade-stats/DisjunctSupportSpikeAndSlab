import numpy
# import autograd.numpy as numpy  # Thinly-wrapped numpy
# from autograd import value_and_grad    # The only autograd function you may ever need
# from autograd import grad
#from autograd.differential_operators import jacobian


import scipy.stats
import scipy.special
import shared.statHelper as statHelper
import scipy.optimize
import shared.analyzeHelper as analyzeHelper
import time
from collections import defaultdict

import pickle

import showResultsText

import rpy2
print("rpy2 version = " + rpy2.__version__)
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
ro.r('source(\'imports.R\')')
ro.r('set.seed(8985331)') # set a fixed random seed to make results of glasso, stars etc reproducible 

# import warnings
# import scipy.linalg

# from SpikeAndSlabNonContinuous_MCMC import SpikeAndSlabProposed as SpikeAndSlabProposed_nonContinuousMCMC
import samplingHelper



class SpikeAndSlabProposedModelSearch:
    
    allDelta = [0.8, 0.5, 0.05, 0.01, 0.001, 0.0]
    
    # hyper-parameters for response variance
    ETA_SQUARE_R = 1.0
    NU_R = 1.0
    
    # hyper-parameters for the relevant regression coefficients
    ETA_SQUARE_1 = 100.0
    NU_1 = 1.0
    
    
    def __init__(self, allObservedResponses, allObservedCovariates, delta):
        
        self.delta = delta
        
        self.nu1 = SpikeAndSlabProposedModelSearch.NU_1
        self.etaSquare1 = SpikeAndSlabProposedModelSearch.ETA_SQUARE_1
        
        if self.delta > 0.0:
                
            deltaToSigmaSquare0 = pickle.load(open( "deltaToSigmaSquare0", "rb" ) )
            self.sigmaSquare0 = deltaToSigmaSquare0[delta]
        else:
            self.sigmaSquare0 = None
        
        self.fullp = allObservedCovariates.shape[1]
        self.y = allObservedResponses
        self.n = allObservedResponses.shape[0]
        
        self.setX(allObservedCovariates)
        
        return
    
    
    @staticmethod
    def getSigmaSquareR_reducedModel(allObservedResponses, allObservedCovariates, delta, selectedVars, NUMBER_OF_MCMC_SAMPLES_TOTAL):
        allObservedCovariates_sub = allObservedCovariates[:, selectedVars]
        subModel = SpikeAndSlabProposedModelSearch(allObservedResponses, allObservedCovariates_sub, delta)
        return subModel.getSigmaSquareR_fullModel_fromCurrentModel(NUMBER_OF_MCMC_SAMPLES_TOTAL)
        
    
    
    # checked
    # calculates p(beta = delta | relevantVariable, nu1, etaSquare1)
    @staticmethod
    def getBetaGivenNu1EtaSquare1(delta, NUMBER_OF_SAMPLES):
        
        allLogProbs = numpy.zeros(NUMBER_OF_SAMPLES)
        for i in range(NUMBER_OF_SAMPLES):
            sigmaSquare = samplingHelper.getScaledInvChiSquareSample(nu = SpikeAndSlabProposedModelSearch.NU_1, etaSquare = SpikeAndSlabProposedModelSearch.ETA_SQUARE_1, numberOfSamples = 1)[0]
            allLogProbs[i] = - ((delta ** 2) / (2.0 * sigmaSquare)) - samplingHelper.exactLogNormalizationConstant_O_static(delta, sigmaSquare)
            
        logProbEstimate = scipy.special.logsumexp(allLogProbs) - numpy.log(NUMBER_OF_SAMPLES)
        
        return logProbEstimate
    
    
    @staticmethod
    def getSigmaSquare0(delta, NUMBER_OF_SAMPLES):
        sigmaSquare0 = 1.0
        logProbDelta_relevant = SpikeAndSlabProposedModelSearch.getBetaGivenNu1EtaSquare1(delta, NUMBER_OF_SAMPLES)
        logProbAtDelta_notRelevant = samplingHelper.truncatedNormal_not_relevant_logDensity(delta, sigmaSquare0, delta)
        
        print("logProbDelta_relevant = ", logProbDelta_relevant)
        print("logProbAtDelta_notRelevant = ", logProbAtDelta_notRelevant)
        
        ERROR_TOLERANCE = 0.001
        previousDirectionDown = True
        assert(logProbAtDelta_notRelevant > logProbDelta_relevant)
        
        base = 1.0
        iterations = 0
        
        while numpy.abs(logProbDelta_relevant - logProbAtDelta_notRelevant) > ERROR_TOLERANCE:
            if logProbAtDelta_notRelevant > logProbDelta_relevant:
                if not previousDirectionDown:
                    base = base / 2.0
                previousDirectionDown = True
                sigmaSquare0 = sigmaSquare0 / (1.0 + base)
            else:
                if previousDirectionDown:
                    base = base / 2.0
                previousDirectionDown = False
                sigmaSquare0 = sigmaSquare0 * (1.0 + base)
            logProbAtDelta_notRelevant = samplingHelper.truncatedNormal_not_relevant_logDensity(delta, sigmaSquare0, delta)
            iterations += 1
            if iterations >= 1000:
                print("ERROR: DID NOT CONVERGE !")
                assert(False)
        
        print("FINAL ESTIMATE:")
        print("logProbDelta_relevant = ", logProbDelta_relevant)
        print("logProbAtDelta_notRelevant = ", logProbAtDelta_notRelevant)
        print("sigmaSquare0 = ", sigmaSquare0)
        return sigmaSquare0
    
    def setX(self, allObservedCovariates):
        self.p = allObservedCovariates.shape[1]
        assert(allObservedCovariates.shape[0] == self.n)
        
        self.X = allObservedCovariates
        
        self.XTX = numpy.matmul(self.X.transpose(), self.X)
        self.yX = numpy.matmul(self.y.transpose(), self.X)
        self.invXTX_regularized = numpy.linalg.inv(self.XTX + numpy.eye(self.p))
        return
    
    # checked2
    def sampleFullBeta(self, z, sigmaSquareR, slabVariance):
        
        fullBeta = numpy.zeros(self.p)
        
        if numpy.sum(z) == 0:
            return fullBeta
        
        fullX = self.X[:, z == 1]
        mu, sigma, _ = SpikeAndSlabProposedModelSearch.getFullNormalParameters(self.y, fullX, sigmaSquareR, slabVariance)
        
        betaNonZero = numpy.random.multivariate_normal(mu, sigma)
        
        fullBeta[z == 1] = betaNonZero
        
        return fullBeta
    
    # checked2
    # gets the parameters of the normal which specifies p(beta | y, fullX, sigmaSquareR, slabVariance)
    @staticmethod
    def getFullNormalParameters(y, fullX, sigmaSquareR, slabVariance):
        invSigma = (1.0 / sigmaSquareR) * numpy.matmul(fullX.transpose(), fullX) + (1.0 / slabVariance) * numpy.eye(fullX.shape[1])
        sigma = numpy.linalg.inv(invSigma)
        mu = (1.0 / sigmaSquareR) * numpy.matmul( numpy.matmul(y.transpose(), fullX), sigma)
        return mu, sigma, invSigma
    
    
    # reading checked2
    # returns the mean and variance of p(beta_{s+1} | beta_1, ..., beta_s, sigmaSquareR, slabVariance)
    # j specifies the index of the new beta coefficient (i.e. corresponds to s+1)
    # conditionedBeta corresponds to beta_1, ..., beta_s
    def getNormalParamertersForU(self, z, j, fullBeta, sigmaSquareR, slabVariance):
        
        zWithoutJ = numpy.copy(z)
        zWithoutJ[j] = 0
        
        conditionedBeta = fullBeta[zWithoutJ == 1]
        sPlusOneIndicies = numpy.where(zWithoutJ == 1)[0]
        sPlusOneIndicies = numpy.append([j], sPlusOneIndicies)
        
        s = numpy.sum(zWithoutJ)
        assert(conditionedBeta.shape[0] == s)
        assert(len(sPlusOneIndicies) == s + 1)
        
        fullX = self.X[:, sPlusOneIndicies]
        assert(fullX.shape[1] == len(sPlusOneIndicies))
        mu, sigma, invSigma = SpikeAndSlabProposedModelSearch.getFullNormalParameters(self.y, fullX, sigmaSquareR, slabVariance)
        
        invSimga22 = numpy.linalg.inv( sigma[1:(s+1), 1:(s+1)] )
        sigma12 = sigma[0,1:(s+1)]
        
        tmpVec = numpy.matmul(sigma12, invSimga22)
        newMean = mu[0] + numpy.dot(tmpVec, conditionedBeta - mu[1:(s+1)])
        
        newVariance = 1.0 / invSigma[0,0]
        
        assert(newVariance > 0.0)
        return newMean, newVariance
    
    
    
    # reading checked2
    # calculates log p(beta, sigmaSquareR, sigmaSquare, y | X, S)
    def getJointLogProb_forRJMCMC(self, z, fullBeta, sigmaSquareR, slabVariance):
        assert(sigmaSquareR > 0.0)
        
        restrictedX = self.X[:, z == 1]
        restrictedBeta = fullBeta[z == 1]
        s = restrictedBeta.shape[0]
        assert(s == numpy.sum(z))
        
        jointLogProb = - (float(self.n) / 2.0) * numpy.log(2.0 * numpy.pi)
        jointLogProb -= (((SpikeAndSlabProposedModelSearch.NU_R + self.n) / 2.0) + 1.0) * numpy.log(sigmaSquareR)
        jointLogProb -= (1.0 / (2.0 * sigmaSquareR)) * (SpikeAndSlabProposedModelSearch.NU_R * SpikeAndSlabProposedModelSearch.ETA_SQUARE_R + numpy.sum(numpy.square(self.y - numpy.matmul(restrictedX, restrictedBeta))))
        
        jointLogProb -= s * numpy.log(2.0 * numpy.pi * slabVariance)
        jointLogProb -= (1.0 / (2.0 * slabVariance)) * numpy.sum(numpy.square(restrictedBeta))
        
        # add prior on z
        jointLogProb += SpikeAndSlabProposedModelSearch.getLogPriorZ(s, self.fullp)
        
        return jointLogProb
    
    
    # reading checked2
    # calculates log p(beta, sigmaSquareR, sigmaSquare, y | X, S)
    def getJointLogProb_forSimple(self, z, restrictedBeta, sigmaSquareR, slabVariance):
        assert(sigmaSquareR > 0.0)
        assert(numpy.sum(z) == restrictedBeta.shape[0])
        
        restrictedX = self.X[:, z == 1]
        s = restrictedBeta.shape[0]
        assert(s == numpy.sum(z))
        
        jointLogProb = - (float(self.n) / 2.0) * numpy.log(2.0 * numpy.pi)
        jointLogProb -= (((SpikeAndSlabProposedModelSearch.NU_R + self.n) / 2.0) + 1.0) * numpy.log(sigmaSquareR)
        jointLogProb -= (1.0 / (2.0 * sigmaSquareR)) * (SpikeAndSlabProposedModelSearch.NU_R * SpikeAndSlabProposedModelSearch.ETA_SQUARE_R + numpy.sum(numpy.square(self.y - numpy.matmul(restrictedX, restrictedBeta))))
        
        jointLogProb -= s * numpy.log(2.0 * numpy.pi * slabVariance)
        jointLogProb -= (1.0 / (2.0 * slabVariance)) * numpy.sum(numpy.square(restrictedBeta))
        
        # add prior on z
        jointLogProb += SpikeAndSlabProposedModelSearch.getLogPriorZ(s, self.fullp)
        
        return jointLogProb
    
    
    
    # reading checked2
    # use this if delta = 0.0
#     def sampleZjConditionedOnRest_RJMCMC(self, sigmaSquareR, slabVariance, fullBeta, z, j):
#         
#         zWithoutJ = numpy.copy(z)
#         zWithoutJ[j] = 0
#         logJointProbWithoutJ = self.getJointLogProb_forRJMCMC(zWithoutJ, fullBeta, sigmaSquareR, slabVariance)
#         
#         uMean, uVariance = self.getNormalParamertersForU(z, j, fullBeta, sigmaSquareR, slabVariance)
#     
#         if z[j] == 1:
#             # try to traverse to model with z[j] = 0
#             logJointProbWithJ = self.getJointLogProb_forRJMCMC(z, fullBeta, sigmaSquareR, slabVariance)
#             logG_toHigherDimension = scipy.stats.norm.logpdf(fullBeta[j], loc=uMean, scale=numpy.sqrt(uVariance))
#             
#             logRatio = (logJointProbWithoutJ + logG_toHigherDimension) - logJointProbWithJ
#             
#             self.totalSamplingCount_RJMCMC += 1.0
#             self.totalAcceptanceRate_RJMCMC += numpy.min([1.0, numpy.exp(logRatio)])
#             
#             if numpy.random.uniform() < numpy.exp(logRatio):
#                 return 0
#             else:
#                 return 1
#             
#         else:
#             # try to traverse to model with z[j] = 1
#             fullBeta[j] = scipy.stats.norm.rvs(loc=uMean, scale=numpy.sqrt(uVariance))
#             logG_toHigherDimension = scipy.stats.norm.logpdf(fullBeta[j], loc=uMean, scale=numpy.sqrt(uVariance))
#             
#             zWithJ = numpy.copy(z)
#             zWithJ[j] = 1
#             logJointProbWithJ = self.getJointLogProb_forRJMCMC(zWithJ, fullBeta, sigmaSquareR, slabVariance)
#             
#             logRatio = logJointProbWithJ - (logJointProbWithoutJ + logG_toHigherDimension) 
#             
#             self.totalSamplingCount_RJMCMC += 1.0
#             self.totalAcceptanceRate_RJMCMC += numpy.min([1.0, numpy.exp(logRatio)])
#             
#             if numpy.random.uniform() < numpy.exp(logRatio):
#                 return 1
#             else:
#                 return 0
            
    
    
    # returns p(z, sigmaSquareR, slabVariance, y)
#     def getJointLogProbZ_Sigma_y(self, sigmaSquareR, slabVariance, z):
#         assert(self.delta == 0.0)
#         
#         # print("z = ", z)
#         # print("self.X.shape = ", self.X.shape)
#         
#         restrictedX = self.X[:, z == 1]
#         mu, _, invSigma = SpikeAndSlabProposedModelSearch.getFullNormalParameters(self.y, restrictedX, sigmaSquareR, slabVariance)
#         
#         jointLogProb = self.getJointLogProb_forSimple(z, mu, sigmaSquareR, slabVariance)
#         posteriorLogProb = -0.5 * (invSigma.shape[0] * numpy.log(2.0 * numpy.pi) - idcHelper.getLogDet(invSigma))
#         
#         # posteriorLogProb = scipy.stats.multivariate_normal.logpdf(mu, mean=mu, cov=sigma)        
# #         print("mu = ", mu)
#         # print("posteriorLogProb = ", posteriorLogProb)
# #         posteriorLogProb = -0.5 * (sigma.shape[0] * numpy.log(2.0 * numpy.pi) - idcHelper.getLogDet(invSigma))
# #         print("posteriorLogProb = ", posteriorLogProb)
#         # assert(False)
#         
#         return jointLogProb - posteriorLogProb
        
    
    
#     def sampleZjConditionedOnRest_delta0_simple(self, sigmaSquareR, slabVariance, z, j):
#         
#         unnormalizedLogProbZ = numpy.zeros(2)
#         
#         zWithoutJ = numpy.copy(z)
#         zWithoutJ[j] = 0
#         unnormalizedLogProbZ[0] = self.getJointLogProbZ_Sigma_y(sigmaSquareR, slabVariance, zWithoutJ)
#         
#         zWithJ = numpy.copy(z)
#         zWithJ[j] = 1
#         unnormalizedLogProbZ[1] = self.getJointLogProbZ_Sigma_y(sigmaSquareR, slabVariance, zWithJ)
#         
#         logNormalization = scipy.special.logsumexp(unnormalizedLogProbZ)
#         zProbs = numpy.exp(unnormalizedLogProbZ - logNormalization)
#         newZj = numpy.random.choice(numpy.arange(2), p=zProbs)
#         
#         return newZj
    
    
    
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
                beta[j] = SpikeAndSlabProposedModelSearch.sampleTruncatedBeta(self.delta, meanTilde, sigmaSquareTilde, z[j] == 1)
            

            if self.delta == 0:
                # safety check for delta == 0
                assert(numpy.all(beta[z == 0] == 0) and numpy.all(beta[z == 1] != 0))
                
            
            # sample p(sigmaSquareR | beta, z, y, X)
            etaSquareForsigmaSquareR = (SpikeAndSlabProposedModelSearch.NU_R * SpikeAndSlabProposedModelSearch.ETA_SQUARE_R + numpy.sum(numpy.square(self.y - numpy.matmul(self.X, beta)))) / (SpikeAndSlabProposedModelSearch.NU_R + self.n)
            sigmaSquareR = samplingHelper.getScaledInvChiSquareSample(nu = SpikeAndSlabProposedModelSearch.NU_R + self.n, etaSquare = etaSquareForsigmaSquareR, numberOfSamples = 1)[0]
            
            # sample p(sigmaSquare_0 | beta, z, y, X) and p(sigmaSquare_1 | beta, z, y, X)
            spikeAndSlabVar[1] = self.sampleSigmaSquareConditional(True, beta, z)
            
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
                
                unnormalizedLogProbZ[sspInd] += SpikeAndSlabProposedModelSearch.getTruncatedNormalLogConstant(sspInd, self.delta, sigmaSquareTilde, meanTilde)
                unnormalizedLogProbZ[sspInd] -= SpikeAndSlabProposedModelSearch.getTruncatedNormalLogConstant(sspInd, self.delta, spikeAndSlabVar[sspInd], 0.0)
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
    
    
    # BRAND-NEW CHECKED
    @staticmethod
    def getTruncatedNormalLogConstant(sspIndicator, delta, sigmaSquare, mean):
        assert(sspIndicator == 0 or sspIndicator == 1)
        if sspIndicator == 1:
            return samplingHelper.exactLogNormalizationConstant_O_static(delta, sigmaSquare, mean)
        else:
            # print("sigmaSquare = ", str(sigmaSquare) + ", mean = " + str(mean))
            return samplingHelper.exactLogNormalizationConstant_I_static(delta, sigmaSquare, mean)
    
    
    # BRAND-NEW CHECKED
    # sample p(beta_j | beta_-j, z, y, sigmaSquareR, X)
    @staticmethod
    def sampleTruncatedBeta(delta, mean, sigmaSquare, relevant):
        if relevant:
            # RELEVANT
            if delta == 0.0:
                return scipy.stats.norm.rvs(loc=mean, scale=numpy.sqrt(sigmaSquare))
                
            # newBetaJ = sampleTruncatedNormalNaive(True, mean, sigmaSquareTilde)
            newBetaJ = SpikeAndSlabProposedModelSearch.sampleTruncatedNormalAdvanced_outerInterval(delta, mean, sigmaSquare)
            # print("j = ", j)
            # print("newBetaJ = ", newBetaJ)
            assert(newBetaJ <= -delta or newBetaJ >= delta)
            return newBetaJ
        else:
            # NOT RELEVANT
            if delta == 0.0:
                return 0.0
            
            ro.globalenv['sd'] = numpy.sqrt(sigmaSquare)
            ro.globalenv['mean'] = mean
            ro.globalenv['a'] = -delta
            ro.globalenv['b'] = delta
            newBetaJ = ro.r('rtruncnorm(n = 1, a=a , b=b, mean = mean, sd = sd)')[0]
            assert(newBetaJ >= -delta and newBetaJ <= delta)
            return newBetaJ
    
    
    
    # must check whether this is really correct !
    @staticmethod
    def sampleTruncatedNormalAdvanced_outerInterval(delta, mean, sigmaSquare):
        
        # p(beta < -delta)
        lowerBoundIntegral = scipy.stats.norm.logcdf(-delta, loc=mean, scale=numpy.sqrt(sigmaSquare))
        
        # p(beta > delta)
        upperBoundIntegral = scipy.stats.norm.logsf(delta, loc=mean, scale=numpy.sqrt(sigmaSquare))
        
        normalization = scipy.special.logsumexp([lowerBoundIntegral, upperBoundIntegral])
        pLowerProb = numpy.exp(lowerBoundIntegral - normalization)
        pUpperProb = numpy.exp(upperBoundIntegral - normalization)
        
        # print("pLowerProb = ", pLowerProb)
        # print("pUpperProb = ", pUpperProb)
        
        rndUniform = scipy.stats.uniform.rvs()
        if rndUniform < pLowerProb:
            ro.globalenv['a'] = float("-inf")
            ro.globalenv['b'] = -delta
        else:
            ro.globalenv['a'] = delta
            ro.globalenv['b'] = float("+inf")
            
        ro.globalenv['sd'] = numpy.sqrt(sigmaSquare)
        ro.globalenv['mean'] = mean
        
        # print("ro.globalenv['a'] = ", ro.globalenv['a'])
        # print("ro.globalenv['b'] = ", ro.globalenv['b'])
        # print("ro.globalenv['sd'] = ", ro.globalenv['sd'])
        # print("ro.globalenv['mean'] = ", ro.globalenv['mean'])
        
        newBetaJ = ro.r('rtruncnorm(n = 1, a=a , b=b, mean = mean, sd = sd)')[0]
        
        return newBetaJ
    
    
    # REVISED
    # reading checked + experiment check
    # samples from p(sigma_j^2 | beta_j, y, X, S)
    # a slice sampler as in Bayesian Methods for Data Analysis, Carlin et al Third Edition, page 139
    def sampleSigmaSquareConditional(self, relevantVariable, beta, z):
          
        if relevantVariable:
            # SUFFIENTLY GOOD !!
            # sigma >> 1 and delta << 1
            usedBetaCount = numpy.sum(z)
            betaSquareSum = numpy.sum(numpy.square(beta[z == 1]))
            
            etaSquarePrior = self.etaSquare1
            priorNu = self.nu1
            assumeTruncatedNorm_NotRelevant_isConstant = None
            nu = priorNu + usedBetaCount
            
            etaSquare = (priorNu * etaSquarePrior + betaSquareSum) / nu
        else:
            # SUFFIENTLY GOOD !!
            # sigma << 1 and interval is I
            betaSquareSum = numpy.sum(numpy.square(beta[z == 0]))
            etaSquarePrior = self.etaSquare0
            priorNu = self.nu0
            
            usedBetaCount = self.p - numpy.sum(z)
            
            if self.delta >= 0.01:
                assumeTruncatedNorm_NotRelevant_isConstant = False
                nu = priorNu + usedBetaCount
                
                etaSquare = (priorNu * etaSquarePrior + betaSquareSum) / nu
            else:
                assumeTruncatedNorm_NotRelevant_isConstant = True
                nu = priorNu
                # nu = self.nu
                # etaSquare = (self.nu * self.etaSquare0 + singleBeta ** 2 - self.delta ** 2) / self.nu
        
                etaSquare = (priorNu * etaSquarePrior + betaSquareSum - usedBetaCount * ((self.delta / 2.0) ** 2)) / nu
        
        # initialize with mode
        sigmaSquare = (nu * etaSquare) / (nu + 2)
        assert(sigmaSquare > 0.0)
        
        numberOfSamples = 1
        
        BURN_IN = 10
        acceptanceCount = 0  # only for checking acceptance ratio
        nrIterations = 0
        acquiredSamples = []
        while len(acquiredSamples) < numberOfSamples:
            nrIterations += 1
            u = scipy.stats.uniform.rvs(loc=0, scale=1, size=1)[0] * self.h(relevantVariable, sigmaSquare, usedBetaCount, assumeTruncatedNorm_NotRelevant_isConstant)
            newSigmaSquare = samplingHelper.getScaledInvChiSquareSample(nu, etaSquare, 1)[0]
            
            if nrIterations >= 2000:
                print("WARNING QUIT WITHOUT PROPER ACCEPTANCE")
                print("relevantVariable = ", relevantVariable)
                print("mode  = ", ( (nu * etaSquare) / (nu + 2)) )
                print("usedBetaCount = ", usedBetaCount)
                acquiredSamples.append(newSigmaSquare)
                break
            
            if u < self.h(relevantVariable, newSigmaSquare, usedBetaCount, assumeTruncatedNorm_NotRelevant_isConstant):
                acceptanceCount += 1
                sigmaSquare = newSigmaSquare
                if acceptanceCount >= BURN_IN:
                    acquiredSamples.append(newSigmaSquare)
        
        
        
        # assert(acceptanceRatio > 0.01) # should be larger than 1% (if numberOfSamples = 1, then by chance we might have sometimes low acceptance ratios of  the first 100.)
        
        # assert(acceptanceRatio > 0.0)
        # if (acceptanceRatio <= 0.2):
        #    print("relevantVariable = ", relevantVariable)
        #    print("singleBeta = ", singleBeta)
        # acceptanceRatio = (acceptanceCount / nrIterations)
        # print("nrIterations = ", nrIterations)
        # print("acceptance ratio = ", acceptanceRatio)
        # assert(False)
        
        # print("acquiredSamples = ", acquiredSamples)
        assert(len(acquiredSamples) == 1)
        
        return acquiredSamples[0]
    
        
    
    # REVISED
    # reading checked
    def h(self, relevantVariable, sigmaSquare, usedBetaCount, assumeTruncatedNorm_NotRelevant_isConstant):
        assert(sigmaSquare > 0.0)
        assert(numpy.isscalar(sigmaSquare))
        
        if relevantVariable:
            # ALWAYS WINNER
            # sigma >> 1 and interval is O
            return numpy.exp( usedBetaCount *  (numpy.log(numpy.sqrt(2.0 * numpy.pi * sigmaSquare)) - samplingHelper.exactLogNormalizationConstant_O_static(self.delta, sigmaSquare)))
        else:
            # sigma << 1 and interval is I
            if assumeTruncatedNorm_NotRelevant_isConstant:
                return numpy.exp( usedBetaCount *  (- ((self.delta / 2) ** 2) / (2.0 * sigmaSquare) - samplingHelper.exactLogNormalizationConstant_I_static(self.delta, sigmaSquare)))
                
                # assumes that Z(N, sigma) is roughly constant
                # return numpy.exp( - (usedBetaCount * SpikeAndSlabProposed_nonContinuousMCMC.exactLogNormalizationConstant_I_static(self.delta, sigmaSquare)))
            else:
                # assumes that   sqrt(2.0 * numpy.pi * sigmaSquare)  / Z(N, sigma)   is roughly constant
                return numpy.exp( usedBetaCount *  (numpy.log( numpy.sqrt(2.0 * numpy.pi * sigmaSquare)  )  - samplingHelper.exactLogNormalizationConstant_I_static(self.delta, sigmaSquare)))
            
    





    # *****************************************************************
    # ********** METHODS FOR MARGINAL LIKELIHOOD ESTIMATION ***********
    # *****************************************************************

    @staticmethod
    def truncateToValidBeta(delta, z, beta):
        assert(z.shape[0] == beta.shape[0])
        
        truncatedBeta = numpy.copy(beta)
        
        for j in range(beta.shape[0]):
            if z[j] == 1:
                if (beta[j] > - delta) and (beta[j] < delta):
                    if beta[j] <= 0.0:
                        truncatedBeta[j] = -delta
                    else:
                        truncatedBeta[j] = delta
                
            else:
                
                if (beta[j] < -delta) or (beta[j] > delta):
                    if beta[j] <= 0.0:
                        truncatedBeta[j] = -delta
                    else:
                        truncatedBeta[j] = delta
            
        return truncatedBeta
    
    
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
    
    
    def getSigmaSquareR_fullModel_fromCurrentModel(self, NUMBER_OF_MCMC_SAMPLES_TOTAL):
        z = numpy.ones(self.p, dtype = numpy.int)
        
        numberOfFreeBeta = self.p
        fixedBetaPart = numpy.zeros(self.p - numberOfFreeBeta)
        fixedSigmaSquareR = None
        fixedSlabVar = None
        posteriorBeta, posteriorSigmaSquareR, posteriorSlabVar = self.posteriorParameterSamples(z, NUMBER_OF_MCMC_SAMPLES_TOTAL, fixedSlabVar, fixedSigmaSquareR, numberOfFreeBeta, fixedBetaPart)
        
        return numpy.mean(posteriorSigmaSquareR)
        
        
    # z is always considered fixed
    def posteriorParameterSamples(self, z, NUMBER_OF_MCMC_SAMPLES_TOTAL, fixedSlabVar, fixedSigmaSquareR, numberOfFreeBeta, fixedBetaPart):
        assert(fixedSlabVar is None or fixedSlabVar > 0.0)
        assert(fixedSigmaSquareR is None or fixedSigmaSquareR > 0.0)
        assert(numberOfFreeBeta + fixedBetaPart.shape[0] == self.p)
        
        invEst = numpy.linalg.inv(self.X.transpose() @ self.X + 1.0 * numpy.eye(self.p))
        ridgeBetaEst = (invEst @ self.X.transpose()) @ self.y
        
        beta = SpikeAndSlabProposedModelSearch.truncateToValidBeta(self.delta, z, ridgeBetaEst)
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
                beta[j] = SpikeAndSlabProposedModelSearch.sampleTruncatedBeta(self.delta, meanTilde, sigmaSquareTilde, z[j] == 1)
            
            
            if fixedSigmaSquareR is None:
                # sample p(sigmaSquareR | beta, z, y, X)
                etaSquareForsigmaSquareR = (SpikeAndSlabProposedModelSearch.NU_R * SpikeAndSlabProposedModelSearch.ETA_SQUARE_R + numpy.sum(numpy.square(self.y - numpy.matmul(self.X, beta)))) / (SpikeAndSlabProposedModelSearch.NU_R + self.n)
                sigmaSquareR = samplingHelper.getScaledInvChiSquareSample(nu = SpikeAndSlabProposedModelSearch.NU_R + self.n, etaSquare = etaSquareForsigmaSquareR, numberOfSamples = 1)[0]
                
            if fixedSlabVar is None:
                # sample p(sigmaSquare_1 | beta, z, y, X)
                spikeAndSlabVar[1] = self.sampleSigmaSquareConditional(True, beta, z)
            
            # print("spikeAndSlabVar = ", spikeAndSlabVar)
            
            if mcmcIt >= BURN_IN_SAMPLES:
                posteriorBeta[mcmcIt - BURN_IN_SAMPLES] = beta
                posteriorSigmaSquareR[mcmcIt - BURN_IN_SAMPLES] = sigmaSquareR
                posteriorSlabVar[mcmcIt - BURN_IN_SAMPLES] = spikeAndSlabVar[1]
        
        return posteriorBeta, posteriorSigmaSquareR, posteriorSlabVar
    
    
    
    # REVISED
    @staticmethod
    def truncatedBetaLogDensity(betaJVal, delta, mean, sigmaSquare, relevant):
        if relevant:
            if not (betaJVal <= -delta or betaJVal >= delta):
                print("!!!! ERROR HERE !!!!")
                print("delta = ", delta)
                print("betaJVal = ", betaJVal)
            assert(betaJVal <= -delta or betaJVal >= delta)
            logProb = - samplingHelper.exactLogNormalizationConstant_O_static(delta, sigmaSquare, mean)
        else:
            assert(betaJVal >= -delta and betaJVal <= delta)
            logProb = - samplingHelper.exactLogNormalizationConstant_I_static(delta, sigmaSquare, mean)
        
        
        if isinstance(logProb, numpy.ndarray):
            # ensure that it is not an array
            assert(len(logProb) == 1)
            logProb = logProb[0]
        
        logProb -= 0.5 * (1.0 / sigmaSquare) * ( (betaJVal - mean)**2)
        
        assert(logProb > float("-inf") and logProb < float("inf"))
        assert(not numpy.isnan(logProb))
        return logProb
    
    
    def getLogProbBetaJGivenRest(self, sigmaSquareR, spikeAndSlabVar, beta, z, j):
        meanTilde, sigmaSquareTilde, _ =  self.getMeanAndVarOfBetaConditional(sigmaSquareR, spikeAndSlabVar, beta, z, j)
        return SpikeAndSlabProposedModelSearch.truncatedBetaLogDensity(beta[j], self.delta, meanTilde, sigmaSquareTilde, z[j] == 1)
    
    # REVISED
    # calculates log p(beta, sigmaSquareR, sigmaSquare, y | X, S)
    def getJointLogProb_forMarginalCalculation(self, z, beta, sigmaSquareR, spikeAndSlabVar, checkValidBeta = True):
        assert(sigmaSquareR > 0.0)
        
        jointLogProb = - (float(self.n) / 2.0) * numpy.log(2.0 * numpy.pi)
        jointLogProb -= (((SpikeAndSlabProposedModelSearch.NU_R + self.n) / 2.0) + 1.0) * numpy.log(sigmaSquareR)
        jointLogProb -= (1.0 / (2.0 * sigmaSquareR)) * (SpikeAndSlabProposedModelSearch.NU_R * SpikeAndSlabProposedModelSearch.ETA_SQUARE_R + numpy.sum(numpy.square(self.y - numpy.matmul(self.X, beta))))
        jointLogProb -= SpikeAndSlabProposedModelSearch.getScaledInvChiSquareLogNormalizer(SpikeAndSlabProposedModelSearch.ETA_SQUARE_R, SpikeAndSlabProposedModelSearch.NU_R)
        
        for j in range(self.p):
            priorSigmaSquare = spikeAndSlabVar[z[j]]
            assert(priorSigmaSquare > 0.0)
            
            if z[j] == 1:
                if checkValidBeta and ((beta[j] > -self.delta) and (beta[j] < self.delta)):
                    return float("-inf")
                jointLogProb -= samplingHelper.exactLogNormalizationConstant_O_static(self.delta, priorSigmaSquare)
            else:
                if checkValidBeta and ((beta[j] < -self.delta) or (beta[j] > self.delta)):
                    return float("-inf")
                jointLogProb -= samplingHelper.exactLogNormalizationConstant_I_static(self.delta, priorSigmaSquare)
            
            jointLogProb -= (1.0 / (2.0 * priorSigmaSquare)) * (beta[j] ** 2)
        
        # add prior on simgaSquareRelevant
        # jointLogProb += SpikeAndSlabProposedModelSearch.getScaledInvChiSquareLogDensity(spikeAndSlabVar[1], self.etaSquare1, self.nu1)
        
        # add prior on z
        jointLogProb += SpikeAndSlabProposedModelSearch.getLogPriorZ(numpy.sum(z), self.fullp)
            
        return jointLogProb[0]
    
    @staticmethod
    def getLogPriorZ(s, p):
        a = 1.0
        b = 1.0
        priorLogProb = scipy.special.betaln(a + s, b + p - s) - scipy.special.betaln(a, b)
        return priorLogProb
        
    
    # REVISED
    @staticmethod
    def getScaledInvChiSquareLogNormalizer(etaSquare, nu):
        
        nuHalf = nu / 2.0
        logDensity = nuHalf * numpy.log(etaSquare)
        logDensity += nuHalf * numpy.log(nuHalf) - scipy.special.gammaln(nuHalf)
        
        return -logDensity
    
    # returns p(sigmaSqure | etaSquare, nu)
    @staticmethod
    def getScaledInvChiSquareLogDensity(sigmaSquare, etaSquare, nu):
        logProb = - (1.0 / (2.0 * sigmaSquare)) * (etaSquare * nu)
        logProb -= ((nu / 2.0) + 1.0) * numpy.log(sigmaSquare)
            
        logProb -= SpikeAndSlabProposedModelSearch.getScaledInvChiSquareLogNormalizer(etaSquare, nu)
        return logProb
    
    
    @staticmethod
    def fitScaledInvChiSquare(posteriorSlabVar):
        mean = numpy.mean(posteriorSlabVar)
        var = numpy.var(posteriorSlabVar)
        
        nu = 2 * (mean ** 2) / var + 4
        etaSqure = mean * (nu - 2) / nu
        return nu, etaSqure
    
    
    # CHECKED3
    # calculates marginal p(y | X, sigma_0, \hat{sigma}_1) i.e. it does not(!) integrate over \sigma_1, but uses for simplicity the emperical bayes estimate
    # works for both delta > 0 and delta = 0
    def getMarginalLikelihoodEstimate_CHIB(self, selectedVars, NUMBER_OF_MCMC_SAMPLES_TOTAL, allObservedCovariates):
        
        if self.delta == 0.0:
            assert(allObservedCovariates is not None)
            self.setX(allObservedCovariates[:, selectedVars])
            selectedVars = numpy.arange(self.p)
            # print("selectedVars = ", selectedVars)
            # print("X = ", self.X.shape)
            # assert(False)
            
            
        z = numpy.zeros(self.p, dtype = numpy.int)
        z[selectedVars] = 1
        
        numberOfFreeBeta = self.p
        fixedBetaPart = numpy.zeros(self.p - numberOfFreeBeta)
        fixedSigmaSquareR = None
        fixedSlabVar = None
        
        posteriorBeta, posteriorSigmaSquareR, posteriorSlabVar = self.posteriorParameterSamples(z, NUMBER_OF_MCMC_SAMPLES_TOTAL, fixedSlabVar, fixedSigmaSquareR, numberOfFreeBeta, fixedBetaPart)
        
        beta_hat = numpy.mean(posteriorBeta, axis = 0)
        sigmaSquareR_hat = numpy.mean(posteriorSigmaSquareR)
        slabVar_hat = numpy.mean(posteriorSlabVar)
        
        # slabVarPosterior_nu, slabVarPosterior_etaSquare = SpikeAndSlabProposedModelSearch.fitScaledInvChiSquare(posteriorSlabVar)
        # posteriorSigmaSquareSlabLogProb = SpikeAndSlabProposedModelSearch.getScaledInvChiSquareLogDensity(slabVar_hat, slabVarPosterior_etaSquare, slabVarPosterior_nu)
        
        
        # print("betaHat = ", beta_hat)
        # print("sigmaSquareR_hat = ", sigmaSquareR_hat)
        # print("slabVar_hat = ", slabVar_hat)
        spikeAndSlabVar_hat = numpy.asarray([self.sigmaSquare0, slabVar_hat])
        
        
        
        # get log p(beta | y, sigmaSquareR_hat, slabVar_hat)
        posteriorBetaLogProb = self.getLogProbBetaJGivenRest(sigmaSquareR_hat, spikeAndSlabVar_hat, beta_hat, z, 0)
        
        for numberOfFreeBeta in range(2,self.p+1):
            fixedBetaPart = beta_hat[numberOfFreeBeta:self.p]
            # print("fixedBetaPart = ", fixedBetaPart)
            posteriorBeta, _, _ = self.posteriorParameterSamples(z, NUMBER_OF_MCMC_SAMPLES_TOTAL, slabVar_hat, sigmaSquareR_hat, numberOfFreeBeta, fixedBetaPart)
            
            conditionalProbForJ = numberOfFreeBeta - 1
            # print("numberOfFreeBeta = ", numberOfFreeBeta)
            # print("conditionalProbForJ = ", conditionalProbForJ)
            posteriorBeta[:,conditionalProbForJ] = beta_hat[conditionalProbForJ]
            
            NUMBER_OF_ACTUAL_SAMPLES = posteriorBeta.shape[0]
            allLogProbs = numpy.zeros(NUMBER_OF_ACTUAL_SAMPLES)
            for sampleId in range(NUMBER_OF_ACTUAL_SAMPLES):
                allLogProbs[sampleId] = self.getLogProbBetaJGivenRest(sigmaSquareR_hat, spikeAndSlabVar_hat, posteriorBeta[sampleId], z, conditionalProbForJ)
            posteriorBetaLogProb += scipy.special.logsumexp(allLogProbs) - numpy.log(NUMBER_OF_ACTUAL_SAMPLES)
            
        # print("posteriorBetaLogProb = ", posteriorBetaLogProb)
        
        
        
        # get log p(sigmaSquareR_hat | y, slabVar_hat)
        numberOfFreeBeta = self.p
        fixedBetaPart = numpy.zeros(self.p - numberOfFreeBeta)
        fixedSigmaSquareR = None
        posteriorBeta, _, _ = self.posteriorParameterSamples(z, NUMBER_OF_MCMC_SAMPLES_TOTAL, slabVar_hat, fixedSigmaSquareR, numberOfFreeBeta, fixedBetaPart)
        
        NUMBER_OF_ACTUAL_SAMPLES = posteriorBeta.shape[0]
        allLogProbsSigmaSquareR = numpy.zeros(NUMBER_OF_ACTUAL_SAMPLES)
        for sampleId in range(NUMBER_OF_ACTUAL_SAMPLES):
            # get log p(sigma_R | sigma, beta, y)
            etaSquareForsigmaSquareR = (SpikeAndSlabProposedModelSearch.NU_R * SpikeAndSlabProposedModelSearch.ETA_SQUARE_R + numpy.sum(numpy.square(self.y - numpy.matmul(self.X, posteriorBeta[sampleId])))) / (SpikeAndSlabProposedModelSearch.NU_R + self.n)
            allLogProbsSigmaSquareR[sampleId] = samplingHelper.getScaledInvChiSquareLogProb(sigmaSquare = sigmaSquareR_hat, nu = SpikeAndSlabProposedModelSearch.NU_R + self.n, etaSquare = etaSquareForsigmaSquareR)
        posteriorSigmaSquareRLogProb = scipy.special.logsumexp(allLogProbsSigmaSquareR) - numpy.log(NUMBER_OF_ACTUAL_SAMPLES)
        
        # print("posteriorSigmaSquareRLogProb = ", posteriorSigmaSquareRLogProb)
        
        jointLogProb = self.getJointLogProb_forMarginalCalculation(z, beta_hat, sigmaSquareR_hat, spikeAndSlabVar_hat)
        
        # print("jointLogProb = ", jointLogProb)
         
        logMarginalLikelihood = jointLogProb - (posteriorBetaLogProb + posteriorSigmaSquareRLogProb)
        
        if self.delta == 0.0:
            # make sure that it is not used anymore
            self.p = None
            self.X = None
            self.XTX = None
            self.yX = None
            self.invXTX_regularized = None
        
        return logMarginalLikelihood



    def getNormalApproximation(self, z, beta_hat, sigmaSquareR_hat, spikeAndSlabVar_hat):
        assert(numpy.sum(z) <= self.p)
        
        # estimate hessian
        fixedTau = numpy.zeros(self.p)
        
        if (numpy.sum(z) < self.p):
            fixedTau[z == 0] = 1.0 / spikeAndSlabVar_hat[0]
        else:
            assert(fixedTau[z == 0].shape[0] == 0)
        fixedTau[z == 1] = 1.0 / spikeAndSlabVar_hat[1]
        
        tau_r = 1.0 / sigmaSquareR_hat
        
        fullHessian = tau_r * self.XTX + numpy.diag(fixedTau)
        
        covarianceMatrix = numpy.linalg.inv(fullHessian)
        
        # meanVec = (1.0 / sigmaSquareR_MAP_correct) * numpy.matmul(self.yX,  covarianceMatrix)
        # posteriorLogDensityAtMAP = - 0.5 * ( (beta_MAP_correct - meanVec).transpose() @ fullHessian @ (beta_MAP_correct - meanVec) )
        
        
        posteriorLogDensityAtMean = 0.0
        for j in range(self.p):
            posteriorLogDensityAtMean -= SpikeAndSlabProposedModelSearch.getTruncatedLogNormalizationConstant(self.delta, beta_hat[j], covarianceMatrix[j,j], z[j] == 1)
        
        return posteriorLogDensityAtMean
    
    
    def getMarginalLikelihoodEstimate_Approx(self, selectedVars, NUMBER_OF_MCMC_SAMPLES_TOTAL, allObservedCovariates, laplaceApprox):
        
        if self.delta == 0.0:
            assert(allObservedCovariates is not None)
            self.setX(allObservedCovariates[:, selectedVars])
            selectedVars = numpy.arange(self.p)
            
        z = numpy.zeros(self.p, dtype = numpy.int)
        z[selectedVars] = 1
        
        numberOfFreeBeta = self.p
        fixedBetaPart = numpy.zeros(self.p - numberOfFreeBeta)
        fixedSigmaSquareR = None
        fixedSlabVar = None
        
        posteriorBeta, posteriorSigmaSquareR, posteriorSlabVar = self.posteriorParameterSamples(z, NUMBER_OF_MCMC_SAMPLES_TOTAL, fixedSlabVar, fixedSigmaSquareR, numberOfFreeBeta, fixedBetaPart)
        
        beta_hat = numpy.mean(posteriorBeta, axis = 0)
        sigmaSquareR_hat = numpy.mean(posteriorSigmaSquareR)
        slabVar_hat = numpy.mean(posteriorSlabVar)
        spikeAndSlabVar_hat = numpy.asarray([self.sigmaSquare0, slabVar_hat])
        
        
        # approximate log p(beta | y, sigmaSquareR_hat, slabVar_hat)
        
        if laplaceApprox:
            print("RUN LAPLACE APPROXIMATION")
            posteriorBetaLogProb = self.getNormalApproximation(z, beta_hat, sigmaSquareR_hat, spikeAndSlabVar_hat)
        else:
            posteriorBetaLogProb = 0.0
            
            for j in range(self.p):
                bestMean, bestSigmaSquare = SpikeAndSlabProposedModelSearch.learnTruncatedNormal(self.delta, z[j] == 1, posteriorBeta[:,j])
                posteriorBetaLogProb += SpikeAndSlabProposedModelSearch.truncatedBetaLogDensity(beta_hat[j], self.delta, bestMean, bestSigmaSquare, z[j] == 1)
        
        
        
        # get log p(sigmaSquareR_hat | y, slabVar_hat)
        numberOfFreeBeta = self.p
        fixedBetaPart = numpy.zeros(self.p - numberOfFreeBeta)
        fixedSigmaSquareR = None
        posteriorBeta, posteriorSigmaSquareR, posteriorSlabVar = self.posteriorParameterSamples(z, NUMBER_OF_MCMC_SAMPLES_TOTAL, slabVar_hat, fixedSigmaSquareR, numberOfFreeBeta, fixedBetaPart)
        
        NUMBER_OF_ACTUAL_SAMPLES = posteriorBeta.shape[0]
        allLogProbsSigmaSquareR = numpy.zeros(NUMBER_OF_ACTUAL_SAMPLES)
        for sampleId in range(NUMBER_OF_ACTUAL_SAMPLES):
            # get log p(sigma_R | sigma, beta, y)
            etaSquareForsigmaSquareR = (SpikeAndSlabProposedModelSearch.NU_R * SpikeAndSlabProposedModelSearch.ETA_SQUARE_R + numpy.sum(numpy.square(self.y - numpy.matmul(self.X, posteriorBeta[sampleId])))) / (SpikeAndSlabProposedModelSearch.NU_R + self.n)
            allLogProbsSigmaSquareR[sampleId] = samplingHelper.getScaledInvChiSquareLogProb(sigmaSquare = sigmaSquareR_hat, nu = SpikeAndSlabProposedModelSearch.NU_R + self.n, etaSquare = etaSquareForsigmaSquareR)
        posteriorSigmaSquareRLogProb = scipy.special.logsumexp(allLogProbsSigmaSquareR) - numpy.log(NUMBER_OF_ACTUAL_SAMPLES)
        
        jointLogProb = self.getJointLogProb_forMarginalCalculation(z, beta_hat, sigmaSquareR_hat, spikeAndSlabVar_hat)
        
        logMarginalLikelihood = jointLogProb - (posteriorBetaLogProb + posteriorSigmaSquareRLogProb)
        
        if self.delta == 0.0:
            # make sure that it is not used anymore
            self.p = None
            self.X = None
            self.XTX = None
            self.yX = None
            self.invXTX_regularized = None
            
        return logMarginalLikelihood
    
    
    # ****************************************
    # method for learning trunctated normal
    # ****************************************
   
    @staticmethod
    def getTruncatedLogNormalizationConstant(delta, mean, sigmaSquare, relevant):
        if relevant:
            logNormalization = samplingHelper.exactLogNormalizationConstant_O_static(delta, sigmaSquare, mean)
        else:
            logNormalization = samplingHelper.exactLogNormalizationConstant_I_static(delta, sigmaSquare, mean)
        
        if isinstance(logNormalization, numpy.ndarray):
            # ensure that it is not an array
            assert(len(logNormalization) == 1)
            logNormalization = logNormalization[0]
        
        return logNormalization
    
    @staticmethod
    def learnTruncatedNormal(delta, relevant, samples):
        
        NUMBER_MC_SAMPLES = 100
        
        n = samples.shape[0]
        
        def gradTau(mean, tau):
            sigmaSquare = 1.0 / tau
            
            allSamples = numpy.zeros(NUMBER_MC_SAMPLES)
            for i in range(NUMBER_MC_SAMPLES):
                x = SpikeAndSlabProposedModelSearch.sampleTruncatedBeta(delta, mean, sigmaSquare, relevant)
                allSamples[i] = numpy.log((x - mean) ** 2)
            gradNormalizationEstimate = scipy.special.logsumexp(allSamples) - numpy.log(NUMBER_MC_SAMPLES)
            gradNormalizationEstimate -= SpikeAndSlabProposedModelSearch.getTruncatedLogNormalizationConstant(delta, mean, sigmaSquare, relevant)
            gradNormalizationEstimate = - 0.5 * numpy.exp(gradNormalizationEstimate)
            
            gradEst =  0.5 * numpy.sum(numpy.square(samples - mean))
            gradEst += n * gradNormalizationEstimate
            return gradEst
        
        def gradMean(mean, tau):
            return - tau * numpy.sum(samples - mean)
        
        
        def getNegJointLogProb_andGrad(paramsVec):
            mean = paramsVec[0]
            tau = paramsVec[1]
            sigmaSquare = 1.0 / tau
            
            funcValue = 0.5 * tau * numpy.sum(numpy.square(samples - mean))
            funcValue += n * SpikeAndSlabProposedModelSearch.getTruncatedLogNormalizationConstant(delta, mean, sigmaSquare, relevant)
            
            gradient = numpy.asarray([gradMean(mean, tau), gradTau(mean, tau)])
            
            # print("funcValue = ", funcValue)
            # print("gradient = ", gradient)
            return (funcValue, gradient)
    
        
        initialParams = [numpy.mean(samples), 1.0 / numpy.var(samples)]
        
        ALMOST_ZERO = 0.000001
        meanBound = [(float("-inf"), float("inf"))]
        tauBound = [(ALMOST_ZERO, float("inf"))]
        allBounds = meanBound + tauBound 
        
        result = scipy.optimize.minimize(getNegJointLogProb_andGrad, initialParams, method='L-BFGS-B', bounds=allBounds, jac=True) 
        
        allParams_map = result["x"]
        bestMean = allParams_map[0]
        bestSigmaSquare = 1.0 / allParams_map[1]
        
        
        return bestMean, bestSigmaSquare
    
    
    @staticmethod
    def runTest():
        delta = 0.1
        mean = -0.1
        sigmaSquare = 5.5
        relevant = True
        
        NUMBER_OF_TEST_SAMPLES = 1000
        samples = numpy.zeros(NUMBER_OF_TEST_SAMPLES)
        for i in range(NUMBER_OF_TEST_SAMPLES):
            samples[i] = SpikeAndSlabProposedModelSearch.sampleTruncatedBeta(delta, mean, sigmaSquare, relevant)
        
        # print("samples = ", samples)
        # assert(False)
        bestMean, bestSigmaSquare = SpikeAndSlabProposedModelSearch.learnTruncatedNormal(delta, relevant, samples)
        
        print("bestMean = ", bestMean)
        print("bestSigmaSquare = ", bestSigmaSquare)

    
    @staticmethod
    def marginalTest():
        import simDataGeneration
        
        dataType = "correlated"
        noiseRatio = 0.06
        n  = 1000
        delta = 0.01
        NUMBER_OF_REPETITIONS = 5
        
        trueBetaWithoutNoiseOrOutlier, allX, allY, trueBeta, correlationMatrix, responseStd = simDataGeneration.getSyntheticData(dataType, noiseRatio, n, NUMBER_OF_REPETITIONS)
        
        myModelSearch = SpikeAndSlabProposedModelSearch(allY[0], allX[0], delta)
        NUMBER_OF_MCMC_SAMPLES_FOR_MARGINAL_ESTIMATE = 100
        selectedVars = []
        logMarginal = myModelSearch.getMarginalLikelihoodEstimate_CHIB(selectedVars, NUMBER_OF_MCMC_SAMPLES_FOR_MARGINAL_ESTIMATE)
        print("logMarginal = ", logMarginal)


    @staticmethod
    def saveSigma0Values():
        deltaToSigmaSquare0 = {}
        for delta in [0.8, 0.5, 0.05, 0.01, 0.001]:
            deltaToSigmaSquare0[delta] = SpikeAndSlabProposedModelSearch.getSigmaSquare0(delta, 100000)
          
        print("deltaToSigmaSquare0 = ", deltaToSigmaSquare0)
        pickle.dump( deltaToSigmaSquare0, open( "deltaToSigmaSquare0", "wb" ) )
        
# SpikeAndSlabProposedModelSearch.marginalTest()
# [0,1, 2] logMarginal =  -36.493045176888565
# [0,1]  logMarginal =  -33.972331077557584
# [0] logMarginal =  -30.520632101361606
# [] logMarginal =  -33.33177281540057
