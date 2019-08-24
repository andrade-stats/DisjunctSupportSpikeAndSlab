
import numpy
import shared.statHelper as statHelper
import shared.idcHelper as idcHelper




def generateData(responseStd, beta, correlationMatrix, nrTrueSamples, nrOutlierSamples):
    
    zeroVec = numpy.zeros(beta.shape[0])
    X = numpy.random.multivariate_normal(zeroVec, correlationMatrix, nrTrueSamples + nrOutlierSamples)
    yTrue = numpy.matmul(X[0:nrTrueSamples], beta)
    
    if nrOutlierSamples > 0:
        assert(False)
        outlierBeta = numpy.zeros(beta.shape[0])
        outlierBeta[beta == 0] = numpy.max(beta)
        yOutlier = numpy.matmul(X[nrTrueSamples:(nrTrueSamples + nrOutlierSamples)], outlierBeta)
        y = numpy.hstack((yTrue, yOutlier))
    else:
        y = yTrue
    
    noise = responseStd * numpy.random.normal(size=nrTrueSamples + nrOutlierSamples)
    y += noise
    
    return X, y

    
def addNoise(trueBeta, noiseRatio):
    if noiseRatio > 0.0:
        assert(noiseRatio == 0.5 or noiseRatio == 0.2)
        
        if trueBeta.shape[0] > 10:
            MIN_VALUE = noiseRatio
            p = trueBeta.shape[0]
            contaminatedNumber = int(p * 0.01)
            nextFreePosition = numpy.max(numpy.where(trueBeta != 0)[0]) + 1
            trueBeta[nextFreePosition : (nextFreePosition + contaminatedNumber)] = numpy.random.uniform(low = -MIN_VALUE, high = MIN_VALUE, size = contaminatedNumber) 
        else:
            MIN_VALUE = noiseRatio
            numberOfZeros = numpy.sum(trueBeta == 0)
            trueBeta[trueBeta == 0] = numpy.random.uniform(low = -MIN_VALUE, high = MIN_VALUE, size = numberOfZeros) 
    
    return trueBeta



# generates data as in "Regression Shrinkage and Selection via the Lasso"
def generateLassoExampleData(exampleType, noiseRatio, lowResponseStd):
    
    if exampleType == "example1":
        trueBeta = numpy.asarray([3, 1.5, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
        responseStd = 3.0
    if exampleType == "exampleOneHuge":
        trueBeta = numpy.asarray([1000.0, 1.5, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
        responseStd = 3.0
    elif exampleType == "example3":
        trueBeta = numpy.asarray([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        responseStd = 2.0
    elif exampleType == "myExample":
        trueBeta = numpy.asarray([3.0, 2.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
        responseStd = 1.0
    
    if lowResponseStd:
        responseStd = responseStd * 0.1
    
    trueBetaWithoutNoiseOrOutlier = numpy.copy(trueBeta)
    trueBeta = addNoise(trueBeta, noiseRatio)
    
    p = trueBeta.shape[0]
    rho = 0.5
    correlationMatrix = numpy.zeros((p,p))
    for i in range(p):
        for j in range(p):
            correlationMatrix[i,j] = rho ** numpy.abs(i-j)
    
    return trueBeta, trueBetaWithoutNoiseOrOutlier, responseStd, correlationMatrix


def generateOrthogonalDataExample(exampleType, nrTrueSamples, nrOutlierSamples, noiseRatio, lowResponseStd):
    assert(exampleType == "example1")
    
    trueBeta = numpy.asarray([3, 1.5, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    responseStd = 3.0
    
    if lowResponseStd:
        responseStd = responseStd * 0.1
        
    trueBetaWithoutNoiseOrOutlier = numpy.copy(trueBeta)
    trueBeta = addNoise(trueBeta, noiseRatio)
    
    p = trueBeta.shape[0]
    correlationMatrix = numpy.eye(p)
    # for i in range(p):
    #    for j in range(p):
    #        correlationMatrix[i,j] = rho ** numpy.abs(i-j)
    
    X, y = generateData(responseStd, trueBeta, correlationMatrix, nrTrueSamples, nrOutlierSamples)
    
    return X, y, trueBeta, trueBetaWithoutNoiseOrOutlier

        
# generates data as in "Extended Bayesian information criteria for model selection with large model spaces"
def generateEBICExampleData(p, nrTrueSamples, nrOutlierSamples, noiseRatio):
    
    responseStd = 1.0
    trueBetaInitial = numpy.asarray([0.0, 1.0, 0.7, 0.5, 0.3, 0.2])
    
    trueBeta = numpy.zeros(p)
    trueBeta[0:trueBetaInitial.shape[0]] = trueBetaInitial
    
    trueBetaWithoutNoiseOrOutlier = numpy.copy(trueBeta)
    trueBeta = addNoise(trueBeta, noiseRatio)
    
    rho = 0.5
    correlationMatrix = numpy.zeros((p,p))
    for i in range(p):
        for j in range(p):
            correlationMatrix[i,j] = rho ** numpy.abs(i-j)
    
    X, y = generateData(responseStd, trueBeta, correlationMatrix, nrTrueSamples, nrOutlierSamples)
    
    return X, y, trueBeta, trueBetaWithoutNoiseOrOutlier


# generates the data as in Section 4 of "The EM Approach to Bayesian Variable Selection" (they use p = 1000 and n = 100)
def generateEMVSExampleData(noiseRatio):
    
    p = 1000
    responseStd = numpy.sqrt(3)
    
    trueBetaInitial = numpy.asarray([3.0,2.0,1.0])
    
    trueBeta = numpy.zeros(p)
    trueBeta[0:trueBetaInitial.shape[0]] = trueBetaInitial
    
    trueBetaWithoutNoiseOrOutlier = numpy.copy(trueBeta)
    trueBeta = addNoise(trueBeta, noiseRatio)
    
    rho = 0.6
    correlationMatrix = numpy.zeros((p,p))
    for i in range(p):
        for j in range(p):
            correlationMatrix[i,j] = rho ** numpy.abs(i-j)
    
    return trueBeta, trueBetaWithoutNoiseOrOutlier, responseStd, correlationMatrix


def highDimOrthogonal(nrTrueSamples, nrOutlierSamples, noiseRatio):
    assert(nrTrueSamples >= 100)
    
    p = 1000
    responseStd = numpy.sqrt(3)
    
    trueBetaInitial = numpy.asarray([3.0,2.0,1.0])
    
    trueBeta = numpy.zeros(p)
    trueBeta[0:trueBetaInitial.shape[0]] = trueBetaInitial
    
    trueBetaWithoutNoiseOrOutlier = numpy.copy(trueBeta)
    trueBeta = addNoise(trueBeta, noiseRatio)
    
    correlationMatrix = numpy.eye(p)
    X, y = generateData(responseStd, trueBeta, correlationMatrix, nrTrueSamples, nrOutlierSamples)
    
    return X, y, trueBeta, trueBetaWithoutNoiseOrOutlier


# checked
def getExpectedMSE(trueBeta, correlationMatrix, responseStd, estimatedBeta):
    mse = responseStd ** 2
    mse += (trueBeta - estimatedBeta) @ correlationMatrix @ (trueBeta - estimatedBeta)
    return mse


def getSyntheticData(dataType, noiseRatio, n, NUMBER_OF_REPETITIONS):
    
    assert(noiseRatio >= 0.0)
    
    RANDOM_GENERATOR_SEED = 9899832
    numpy.random.seed(RANDOM_GENERATOR_SEED)
    
    outlierRatio = 0.0
    lowResponseStd = False
    
    NUMBER_OF_TRUE_MODEL_SAMPLES = int((1.0 - outlierRatio) * n)
    NUMBER_OF_OUTLIER_SAMPLES = int(outlierRatio * n)
    assert(n == NUMBER_OF_TRUE_MODEL_SAMPLES + NUMBER_OF_OUTLIER_SAMPLES)
    
    allX = []
    allY = []
    
    if dataType == "highDim":
        trueBeta, trueBetaWithoutNoiseOrOutlier, responseStd, correlationMatrix = generateEMVSExampleData(noiseRatio)
    elif dataType == "highDimOr":
        trueBeta, trueBetaWithoutNoiseOrOutlier = highDimOrthogonal(NUMBER_OF_TRUE_MODEL_SAMPLES, NUMBER_OF_OUTLIER_SAMPLES, noiseRatio)
    elif dataType == "correlated":
        trueBeta, trueBetaWithoutNoiseOrOutlier, responseStd, correlationMatrix  = generateLassoExampleData("example1", noiseRatio, lowResponseStd)
    elif dataType == "orthogonal":
        trueBeta, trueBetaWithoutNoiseOrOutlier = generateOrthogonalDataExample("example1", NUMBER_OF_TRUE_MODEL_SAMPLES, NUMBER_OF_OUTLIER_SAMPLES, noiseRatio, lowResponseStd)
    elif dataType == "oneHuge":
        trueBeta, trueBetaWithoutNoiseOrOutlier, responseStd, correlationMatrix  = generateLassoExampleData("exampleOneHuge", noiseRatio, lowResponseStd)
    else:
        assert(False)
    
    
    print("trueBeta = ", trueBeta)
    print("trueBetaWithoutNoiseOrOutlier = ", trueBetaWithoutNoiseOrOutlier)
    # print("clean noise rep:")
    # print(", ".join([str(round(x, 2)) for x in trueBeta]))
    bestMSE = getExpectedMSE(trueBeta, correlationMatrix, responseStd, trueBeta)
    simplifiedMSE = getExpectedMSE(trueBeta, correlationMatrix, responseStd, trueBetaWithoutNoiseOrOutlier)
    mseIncrease = (simplifiedMSE / bestMSE) - 1.0
    print("expected MSE of true beta = ", bestMSE)
    print("expected MSE of simplified beta = ", simplifiedMSE)
    print("expected increase in MSE of simplified beta = ", mseIncrease)
        
    for repetitionId in range(NUMBER_OF_REPETITIONS):
        X, y = generateData(responseStd, trueBeta, correlationMatrix, NUMBER_OF_TRUE_MODEL_SAMPLES, NUMBER_OF_OUTLIER_SAMPLES)
        
        # normalize covariates and subtract mean from response so that no bias term is needed
        X = statHelper.normalizeData(X)
        y = y - numpy.mean(y)

        allX.append(X)
        allY.append(y)
    
    return trueBetaWithoutNoiseOrOutlier, allX, allY, trueBeta, correlationMatrix, responseStd
