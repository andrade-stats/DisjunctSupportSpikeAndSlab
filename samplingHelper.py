
import scipy.stats
import numpy
import shared.statHelper as statHelper
import scipy.special


# REVISED
# implements sampling of scaled inverse-chi-square density according to "Bayesian Data Analysis", page 576 denoted as Inv-\chi^2
def getScaledInvChiSquareSample(nu, etaSquare, numberOfSamples):
    samples = scipy.stats.invgamma.rvs(a = nu / 2.0, loc=0, scale= (nu / 2.0) * etaSquare, size = numberOfSamples)
    return samples

# REVISED
def getScaledInvChiSquareLogProb(sigmaSquare, nu, etaSquare):
    logProb = scipy.stats.invgamma.logpdf(x = sigmaSquare, a = nu / 2.0, loc=0, scale= (nu / 2.0) * etaSquare)
    return logProb


# REVISED
# calculates the probability of normal random varible being in the interval [-delta, delta]
# helper for logExactNormalizationConstant_I and logExactNormalizationConstant_0
def getLogProbNormalDeltaInterval(delta, sigmaSquare, mean):
    logUpperBoundIntegral = scipy.stats.norm.logcdf(delta, loc=mean, scale=numpy.sqrt(sigmaSquare))
    logLowerBoundIntegral = scipy.stats.norm.logcdf(-delta, loc=mean, scale=numpy.sqrt(sigmaSquare))
    
    # print("upperBoundIntegral = ", upperBoundIntegral)
    # print("lowerBoundIntegral = ", lowerBoundIntegral)
        
    if numpy.isscalar(sigmaSquare):
        assert(logUpperBoundIntegral > -numpy.inf)
        assert(logLowerBoundIntegral > -numpy.inf)
        assert(logUpperBoundIntegral >= logLowerBoundIntegral)
        
        # single number 
        if logUpperBoundIntegral == 0.0:
            # that means, we can actually ignore the right bound at delta
            return scipy.stats.norm.logsf(-delta, loc=mean, scale=numpy.sqrt(sigmaSquare))
        elif (logUpperBoundIntegral - logLowerBoundIntegral > 500.0):
            # the lowerBoundIntegral value is so small (compared to upper bound) that we can ignore it
            return numpy.asarray([logUpperBoundIntegral])
        elif (logUpperBoundIntegral - logLowerBoundIntegral) < 1.0e-14:
            # difference is too small, because 1 < upperBound / lowerBound < exp(1.0e-14)
            # therefore assume upperBound = lowerBound
            return -numpy.inf
        else:
            result = statHelper.logsubexp(logUpperBoundIntegral, logLowerBoundIntegral)
            if result is None:
                print("INFINITY OCCURRED !!")
                print("Debug info:")
                print("delta = ", delta)
                print("sigmaSquare = ", sigmaSquare)
                print("mean = ", mean)
                print("upperBoundIntegral = ", logUpperBoundIntegral)
                print("lowerBoundIntegral = ", logLowerBoundIntegral)
                assert(False)
            else:
                return result

    else:
        # print(type(sigmaSquare))
        # print(sigmaSquare.shape)
        assert(sigmaSquare.shape[0] > 0)
        
        # assert(False)
        # array 
        # if numpy.any(upperBoundIntegral - lowerBoundIntegral > 500.0):
#             overflowIds = numpy.where(upperBoundIntegral - lowerBoundIntegral > 500.0)[0]
#             if len(overflowIds) > 0:
#                 print(overflowIds)
#                 print(lowerBoundIntegral)
#                 print(upperBoundIntegral)
#                 print("sigmaSquare = ", sigmaSquare)
#                 lowerBoundIntegral[overflowIds] = upperBoundIntegral[overflowIds]
#                 deltaIntervals = statHelper.logsubexp(upperBoundIntegral, lowerBoundIntegral)
#                 
#                 print(overflowIds)
#                 print(deltaIntervals)
#                 assert(False)
        
        
        assert(not numpy.any(logUpperBoundIntegral - logLowerBoundIntegral > 500.0))  # otherwise logsubexp is not stable
        return statHelper.logsubexp(logUpperBoundIntegral, logLowerBoundIntegral)
     
    
# REVISED
# double checked for numerical stability
def exactLogNormalizationConstant_I_static(delta, sigmaSquare, mean = 0):
    logProbNormalDeltaInterval = getLogProbNormalDeltaInterval(delta, sigmaSquare, mean)
    logNormConst = logProbNormalDeltaInterval + numpy.log( numpy.sqrt(2 * numpy.pi * sigmaSquare) )
    # assert(numpy.all(logNormConst > float("-inf")) and numpy.all(logNormConst < float("inf")))
    assert(not numpy.any(numpy.isnan(logNormConst)))
    return logNormConst


# REVISED
# double checked for numerical stability
def exactLogNormalizationConstant_O_static(delta, sigmaSquare, mean = 0):
    lowerIntegral = scipy.stats.norm.logcdf(-delta, loc=mean, scale=numpy.sqrt(sigmaSquare))
    # print("lowerIntegral.shape = ", lowerIntegral.shape)
    upperIntegral = scipy.stats.norm.logsf(delta, loc=mean, scale=numpy.sqrt(sigmaSquare))
    stackedVersion = numpy.vstack((lowerIntegral, upperIntegral))
    outerLogIntegral = scipy.special.logsumexp(stackedVersion, axis = 0)
    
    # print("outerLogIntegral.shape = ", outerLogIntegral.shape)
    # print("[lowerIntegral, upperIntegral] = ", stackedVersion.shape)
    logNormConst = outerLogIntegral + numpy.log(numpy.sqrt(2 * numpy.pi * sigmaSquare))
    assert(numpy.all(logNormConst > float("-inf")) and numpy.all(logNormConst < float("inf")))
    assert(not numpy.any(numpy.isnan(logNormConst)))
    return logNormConst
    
    
    
    
    
    
def truncatedNormal_relevant_logDensity(delta, variance, x):
    logDensity = - 0.5 * (1.0 / variance) * (x ** 2)
    logDensity -= exactLogNormalizationConstant_O_static(delta, variance)
    return logDensity
    
    
def truncatedNormal_not_relevant_logDensity(delta, variance, x):
    logDensity = - 0.5 * (1.0 / variance) * (x ** 2)
    logDensity -= exactLogNormalizationConstant_I_static(delta, variance)
    return logDensity
    