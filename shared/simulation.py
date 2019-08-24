import numpy
import shared.idcHelper as idcHelper
import scipy.stats
import networkx

# prepare R 
import rpy2
print("rpy2 version = " + rpy2.__version__)
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
ro.r('source(\'imports.R\')')
ro.r('set.seed(8985331)') # set a fixed random seed to make results of glasso, stars etc reproducible 





def generateData(NUMBER_OF_CLUSTERS, NUMBER_OF_VARIABLES, NUMBER_OF_SAMPLES, clusterSizeDist, sampleType):
    
    assert(False)
    wishartNoiseStdOnPrec = 0.0
    uniformNoiseStdOnPrec = 0.0
    uniformNoiseStdOnCov = 0.0
    invWishartNoiseStdOnCov = 0.0
    invWishartNoiseSCALE = 0.0
        
    if sampleType == "wishartWithWishartNoiseOnPrec":    # used to be noiseWishartOnCov    
        wishartNoiseStdOnPrec = 0.1
        dataVectorsAllOriginal, hiddenVarIds, numberOfClusters, trueCovMatrix, truePrecMatrix = createIndependentDimClusterDataSamples(NUMBER_OF_CLUSTERS, NUMBER_OF_VARIABLES, NUMBER_OF_SAMPLES, wishartNoiseStdOnPrec, uniformNoiseStdOnPrec, uniformNoiseStdOnCov, invWishartNoiseStdOnCov, invWishartNoiseSCALE, clusterSizeDist, "wishart")
        
    elif sampleType == "wishartWithUniformNoiseOnPrec": # used to be noiseUniformOnPrec
        uniformNoiseStdOnPrec = 0.1
        dataVectorsAllOriginal, hiddenVarIds, numberOfClusters, trueCovMatrix, truePrecMatrix = createIndependentDimClusterDataSamples(NUMBER_OF_CLUSTERS, NUMBER_OF_VARIABLES, NUMBER_OF_SAMPLES, wishartNoiseStdOnPrec, uniformNoiseStdOnPrec, uniformNoiseStdOnCov, invWishartNoiseStdOnCov, invWishartNoiseSCALE, clusterSizeDist, "wishart")    
    
    elif sampleType == "wishartWithHighUniformNoiseOnPrec": # used to be noiseUniformOnPrec
        uniformNoiseStdOnPrec = 0.2
        dataVectorsAllOriginal, hiddenVarIds, numberOfClusters, trueCovMatrix, truePrecMatrix = createIndependentDimClusterDataSamples(NUMBER_OF_CLUSTERS, NUMBER_OF_VARIABLES, NUMBER_OF_SAMPLES, wishartNoiseStdOnPrec, uniformNoiseStdOnPrec, uniformNoiseStdOnCov, invWishartNoiseStdOnCov, invWishartNoiseSCALE, clusterSizeDist, "wishart")    
    
    elif sampleType == "uniformSparseWithin" or sampleType == "uniform":
        dataVectorsAllOriginal, hiddenVarIds, numberOfClusters, trueCovMatrix, truePrecMatrix = createIndependentDimClusterDataSamples(NUMBER_OF_CLUSTERS, NUMBER_OF_VARIABLES, NUMBER_OF_SAMPLES, wishartNoiseStdOnPrec, uniformNoiseStdOnPrec, uniformNoiseStdOnCov, invWishartNoiseStdOnCov, invWishartNoiseSCALE, clusterSizeDist, sampleType)
    
    elif sampleType == "uniformWithUniformNoiseOnPrec": # used to be uniformWithUniformNoise
        uniformNoiseStdOnPrec = 0.1
        dataVectorsAllOriginal, hiddenVarIds, numberOfClusters, trueCovMatrix, truePrecMatrix = createIndependentDimClusterDataSamples(NUMBER_OF_CLUSTERS, NUMBER_OF_VARIABLES, NUMBER_OF_SAMPLES, wishartNoiseStdOnPrec, uniformNoiseStdOnPrec, uniformNoiseStdOnCov, invWishartNoiseStdOnCov, invWishartNoiseSCALE, clusterSizeDist, "uniform")
        
    elif sampleType == "uniformWithUniformNoiseOnCov": # used to be uniformWithUniformNoise
        uniformNoiseStdOnCov = 0.01
        dataVectorsAllOriginal, hiddenVarIds, numberOfClusters, trueCovMatrix, truePrecMatrix = createIndependentDimClusterDataSamples(NUMBER_OF_CLUSTERS, NUMBER_OF_VARIABLES, NUMBER_OF_SAMPLES, wishartNoiseStdOnPrec, uniformNoiseStdOnPrec, uniformNoiseStdOnCov, invWishartNoiseStdOnCov, invWishartNoiseSCALE, clusterSizeDist, "uniform")

    elif sampleType == "uniformWithHighUniformNoiseOnPrec": # used to be uniformWithUniformNoise
        uniformNoiseStdOnPrec = 0.2
        dataVectorsAllOriginal, hiddenVarIds, numberOfClusters, trueCovMatrix, truePrecMatrix = createIndependentDimClusterDataSamples(NUMBER_OF_CLUSTERS, NUMBER_OF_VARIABLES, NUMBER_OF_SAMPLES, wishartNoiseStdOnPrec, uniformNoiseStdOnPrec, uniformNoiseStdOnCov, invWishartNoiseStdOnCov, invWishartNoiseSCALE, clusterSizeDist, "uniform")
    elif sampleType == "homogeneous": 
        assert(False)
        # withinClusterCorr = 0.5
        # dataVectorsAllOriginal, hiddenVarIds, numberOfClusters, trueCovMatrix, truePrecMatrix = simulation.createHomogeneousClusters(NUMBER_OF_CLUSTERS, NUMBER_OF_VARIABLES, NUMBER_OF_SAMPLES, withinClusterCorr)

    elif sampleType == "uniformWithWishartNoiseOnPrec": # used to be uniformWithUniformNoise
        wishartNoiseStdOnPrec = 0.001
        dataVectorsAllOriginal, hiddenVarIds, numberOfClusters, trueCovMatrix, truePrecMatrix = createIndependentDimClusterDataSamples(NUMBER_OF_CLUSTERS, NUMBER_OF_VARIABLES, NUMBER_OF_SAMPLES, wishartNoiseStdOnPrec, uniformNoiseStdOnPrec, uniformNoiseStdOnCov, invWishartNoiseStdOnCov, invWishartNoiseSCALE, clusterSizeDist, "uniform")

    elif sampleType == "uniformWithHighWishartNoiseOnPrec": # used to be uniformWithUniformNoise
        wishartNoiseStdOnPrec = 0.01
        dataVectorsAllOriginal, hiddenVarIds, numberOfClusters, trueCovMatrix, truePrecMatrix = createIndependentDimClusterDataSamples(NUMBER_OF_CLUSTERS, NUMBER_OF_VARIABLES, NUMBER_OF_SAMPLES, wishartNoiseStdOnPrec, uniformNoiseStdOnPrec, uniformNoiseStdOnCov, invWishartNoiseStdOnCov, invWishartNoiseSCALE, clusterSizeDist, "uniform")
    
    elif sampleType == "wishart":
        dataVectorsAllOriginal, hiddenVarIds, numberOfClusters, trueCovMatrix, truePrecMatrix = createIndependentDimClusterDataSamples(NUMBER_OF_CLUSTERS, NUMBER_OF_VARIABLES, NUMBER_OF_SAMPLES, wishartNoiseStdOnPrec, uniformNoiseStdOnPrec, uniformNoiseStdOnCov, invWishartNoiseStdOnCov, invWishartNoiseSCALE, clusterSizeDist, "wishart")

    elif sampleType == "invWishartWithUniformNoiseOnPrec":
        uniformNoiseStdOnPrec = 0.1
        dataVectorsAllOriginal, hiddenVarIds, numberOfClusters, trueCovMatrix, truePrecMatrix = createIndependentDimClusterDataSamples(NUMBER_OF_CLUSTERS, NUMBER_OF_VARIABLES, NUMBER_OF_SAMPLES, wishartNoiseStdOnPrec, uniformNoiseStdOnPrec, uniformNoiseStdOnCov, invWishartNoiseStdOnCov, invWishartNoiseSCALE, clusterSizeDist, "inv-wishart")

    elif sampleType == "invWishartWithInvWishartNoiseOnPrec":
        invWishartNoiseSCALE = 100.0
        dataVectorsAllOriginal, hiddenVarIds, numberOfClusters, trueCovMatrix, truePrecMatrix = createIndependentDimClusterDataSamples(NUMBER_OF_CLUSTERS, NUMBER_OF_VARIABLES, NUMBER_OF_SAMPLES, wishartNoiseStdOnPrec, uniformNoiseStdOnPrec, uniformNoiseStdOnCov, invWishartNoiseStdOnCov, invWishartNoiseSCALE, clusterSizeDist, "inv-wishart")

    elif sampleType == "invWishartWithUniformNoiseOnCov":
        uniformNoiseStdOnCov = 0.01
        dataVectorsAllOriginal, hiddenVarIds, numberOfClusters, trueCovMatrix, truePrecMatrix = createIndependentDimClusterDataSamples(NUMBER_OF_CLUSTERS, NUMBER_OF_VARIABLES, NUMBER_OF_SAMPLES, wishartNoiseStdOnPrec, uniformNoiseStdOnPrec, uniformNoiseStdOnCov, invWishartNoiseStdOnCov, invWishartNoiseSCALE, clusterSizeDist, "inv-wishart")

    elif sampleType == "invWishartWithInvWishartNoiseOnCov":
        invWishartNoiseStdOnCov = 0.01
        dataVectorsAllOriginal, hiddenVarIds, numberOfClusters, trueCovMatrix, truePrecMatrix = createIndependentDimClusterDataSamples(NUMBER_OF_CLUSTERS, NUMBER_OF_VARIABLES, NUMBER_OF_SAMPLES, wishartNoiseStdOnPrec, uniformNoiseStdOnPrec, uniformNoiseStdOnCov, invWishartNoiseStdOnCov, invWishartNoiseSCALE, clusterSizeDist, "inv-wishart")

    else:
        assert(sampleType == "invWishart")
        dataVectorsAllOriginal, hiddenVarIds, numberOfClusters, trueCovMatrix, truePrecMatrix = createIndependentDimClusterDataSamples(NUMBER_OF_CLUSTERS, NUMBER_OF_VARIABLES, NUMBER_OF_SAMPLES, wishartNoiseStdOnPrec, uniformNoiseStdOnPrec, uniformNoiseStdOnCov, invWishartNoiseStdOnCov, invWishartNoiseSCALE, clusterSizeDist, "inv-wishart")
    
    return dataVectorsAllOriginal, hiddenVarIds, numberOfClusters, trueCovMatrix, truePrecMatrix

        
# creates a positive semi-definite matrix
def createNoiseCovarianceMatrix(hiddenDataIds, noiseRatio, noiseStrength):
    
    NR_OF_VARIABLES = hiddenDataIds.shape[0]
    noiseCovMatrix = numpy.zeros((NR_OF_VARIABLES,NR_OF_VARIABLES))
    
    randomIdsInOrder = numpy.arange(0, NR_OF_VARIABLES)
            
    for i in range(NR_OF_VARIABLES):
        if numpy.random.rand() < noiseRatio:
            numpy.random.shuffle(randomIdsInOrder)
            sign = numpy.sign(numpy.random.rand() - 0.5)
            for j in range(NR_OF_VARIABLES):
                randomId = randomIdsInOrder[j]
                if hiddenDataIds[randomId] != hiddenDataIds[i]:
                    # add correlation between "randomIds" and "i"
                    noiseCovMatrix[randomId,randomId] += noiseStrength
                    noiseCovMatrix[i,i] += noiseStrength
                    noiseCovMatrix[randomId,i] += sign * noiseStrength
                    noiseCovMatrix[i,randomId] += sign * noiseStrength
                    break
                       
    return noiseCovMatrix





def makePositiveDefinite(matrix):
    NUMBER_OF_VARIABLES = matrix.shape[0]
    eigVals, eigVecs = numpy.linalg.eigh(matrix)
    assert(eigVals[0] < 0.001)
    if eigVals[0] < 0.001:
        # print "eigVals[0] = ", eigVals[0]
        reg = numpy.abs(eigVals[0]) + 0.001
        matrix += reg * numpy.eye(NUMBER_OF_VARIABLES)
    
    return matrix

def normalize(M):
    Lsqrt = numpy.diag(numpy.sqrt(1.0 / numpy.diag(M)))
    normalizedM = Lsqrt @ M @ Lsqrt # @ denotes matrix mulitplication
    return normalizedM


# generate a random correlation as in "Robust Sparse Gaussian Graphical Modeling"
def getUniformRandomEntry(lowerLimit, upperLimit):
    rndSign = (numpy.random.randint(2) * 2) - 1
    assert(rndSign == -1 or rndSign == 1)
    rndMagnitude = numpy.random.uniform(low=lowerLimit, high=upperLimit)
    return rndSign * rndMagnitude


def makeNormalizedPrecisionMatrix(precisionMatrix):
    p = precisionMatrix.shape[0]
    
    # symmetrize    
    precisionMatrix = (precisionMatrix + precisionMatrix.transpose()) / 2.0
    
    # make positive definite
    eigVals, _ = numpy.linalg.eigh(precisionMatrix)
    assert(eigVals[0] < 0)
    reg = 0.1 - eigVals[0]
    assert(reg > 0.0)
    precisionMatrix += reg * numpy.eye(p)
    
    # normalize
    precisionMatrix = normalize(precisionMatrix)
    
    return precisionMatrix

# generates a random precision matrix as in "Robust Sparse Gaussian Graphical Modeling"
def randomBarabasiAlbertPrecisionMatrix(p, parameterOfAttachment, noiseOnPrecisionMatrixRatio):
    assert(noiseOnPrecisionMatrixRatio >= 0.0 and noiseOnPrecisionMatrixRatio <= 1.0)
    
    lowerLimit = 0.25
    upperLimit = 0.75
    
    NOISE_MIN_MAX_RATIO = 10.0
    
    G = networkx.barabasi_albert_graph(p, parameterOfAttachment, 432432)
     
    truePrecisionMatrix = numpy.zeros((p,p))
    
    usedEdgesUpperTriangle = numpy.eye(p)
    usedEdgesUpperTriangle[numpy.tril_indices(p)] = 1.0
    
    trueEdgePositionMatrix = numpy.zeros((p,p))
    
    for e in G.edges():
        truePrecisionMatrix[e] = getUniformRandomEntry(lowerLimit, upperLimit)
        
        assert(e[0] != e[1])
        if e[0] < e[1]:
            usedEdgesUpperTriangle[e[0], e[1]] = 1.0
        else:
            usedEdgesUpperTriangle[e[1], e[0]] = 1.0
    
        trueEdgePositionMatrix[e[0], e[1]] = 1
        trueEdgePositionMatrix[e[1], e[0]] = 1
        
    noisePrecisionMatrix = numpy.copy(truePrecisionMatrix)
    noiseEdgePositionMatrix = numpy.zeros((p,p))
    
    if noiseOnPrecisionMatrixRatio > 0.0:
        numberOfUsedEdges = len(G.edges())
        # print("number of edges in true graph = ", numberOfUsedEdges)
        allZeroIndices = numpy.where(usedEdgesUpperTriangle == 0.0)
        numberOfFreeEdges = allZeroIndices[0].shape[0]
        rndFreeEdgeIds = numpy.arange(numberOfFreeEdges)
        numpy.random.shuffle(rndFreeEdgeIds)
        
        allZeroIndicesX = allZeroIndices[0]
        allZeroIndicesY = allZeroIndices[1]
        
        rndSign = (numpy.random.randint(2, size=numberOfUsedEdges) * 2) - 1
        
        upperNoiseLimit = lowerLimit * noiseOnPrecisionMatrixRatio
        lowerNoiseLimit = upperNoiseLimit / NOISE_MIN_MAX_RATIO
        
        # print("upperNoiseLimit = ", upperNoiseLimit)
        # print("lowerNoiseLimit = ", lowerNoiseLimit)
        
        # add the same number of noise edges as true edges
        for i in range(numberOfUsedEdges):
            rndId = rndFreeEdgeIds[i]
            # noiseEntryHalfRange = lowerLimit * noiseOnPrecisionMatrixRatio
            # noisePrecisionMatrix[allZeroIndicesX[rndId], allZeroIndicesY[rndId]] = numpy.random.uniform(low=-noiseEntryHalfRange, high=noiseEntryHalfRange)
            
            rndNoiseValue = rndSign[i] * numpy.random.uniform(low = lowerNoiseLimit, high = upperNoiseLimit)
            noisePrecisionMatrix[allZeroIndicesX[rndId], allZeroIndicesY[rndId]] = rndNoiseValue
            
            noiseEdgePositionMatrix[allZeroIndicesX[rndId], allZeroIndicesY[rndId]] = 1
            noiseEdgePositionMatrix[allZeroIndicesY[rndId], allZeroIndicesX[rndId]] = 1
            
            
        
        
    # print("precisionMatrix = ")
    # idcHelper.showMatrix(precisionMatrix)
    
    # print("noiseGraph = ")
    # idcHelper.showMatrix(noiseGraph)
    # assert(False)
    
    return makeNormalizedPrecisionMatrix(truePrecisionMatrix), makeNormalizedPrecisionMatrix(noisePrecisionMatrix), trueEdgePositionMatrix, noiseEdgePositionMatrix


def generateDataFromBarabasiAlbertModel(p, n, parameterOfAttachment, noiseOnPrecisionMatrix):
    
    cleanPrecisionMatrix, precisionMatrixWithNoise, trueEdgePositionMatrix, noiseEdgePositionMatrix = randomBarabasiAlbertPrecisionMatrix(p, parameterOfAttachment, noiseOnPrecisionMatrix)
    
#     print("cleanPrecisionMatrix = ")
#     idcHelper.showMatrix(cleanPrecisionMatrix)
#     print("precisionMatrixWithNoise = ")
#     idcHelper.showMatrix(precisionMatrixWithNoise)
#     assert(False)
    
    graphMatrix = numpy.zeros((p,p))
    graphMatrix[cleanPrecisionMatrix != 0] = 1.0
    numpy.fill_diagonal(graphMatrix, 0.0)
    
    trueCovMatrix = numpy.linalg.inv(precisionMatrixWithNoise)
    samples = numpy.random.multivariate_normal(mean = numpy.zeros(p), cov = trueCovMatrix, size = n)
    
    return samples, graphMatrix, cleanPrecisionMatrix, precisionMatrixWithNoise, trueEdgePositionMatrix, noiseEdgePositionMatrix
    
    

def sampleSparseCov(NUMBER_OF_VARIABLES):
    
    precisionMatrix = numpy.zeros((NUMBER_OF_VARIABLES,NUMBER_OF_VARIABLES))
    while True:
        i = numpy.random.randint(low = 0, high = NUMBER_OF_VARIABLES)
        j = numpy.random.randint(low = 0, high = NUMBER_OF_VARIABLES)
        if i != j and precisionMatrix[i,j] == 0.0: 
            precisionMatrix[i,j] = numpy.random.uniform(low = -1.0, high = 1.0)
            precisionMatrix[j,i] = precisionMatrix[i,j]
            if idcHelper.isConnected(precisionMatrix):
                break
      
    assert(idcHelper.isConnected(precisionMatrix)) 
    precisionMatrix = makePositiveDefinite(precisionMatrix)
    covMatrix = numpy.linalg.inv(precisionMatrix)
    corrMatrix = idcHelper.conv2corrMatrix(covMatrix)
    return corrMatrix


def sampleFromInverseWishart(NUMBER_OF_VARIABLES):
    nu0 = NUMBER_OF_VARIABLES + 1
    Sigma0 = numpy.eye(NUMBER_OF_VARIABLES)
    return scipy.stats.invwishart.rvs(df = nu0, scale = Sigma0, size=1)


def sampleFromWishart(NUMBER_OF_VARIABLES):
    precMat = scipy.stats.wishart.rvs(df = NUMBER_OF_VARIABLES + 2, scale = numpy.eye(NUMBER_OF_VARIABLES), size=1)    
    return numpy.linalg.inv(precMat)


def sampleUniformSymmetricMatrix(NUMBER_OF_VARIABLES, alpha):
    symMatrix = numpy.zeros((NUMBER_OF_VARIABLES,NUMBER_OF_VARIABLES))
    for i in range(NUMBER_OF_VARIABLES):
        for j in range(i+1,NUMBER_OF_VARIABLES):
            symMatrix[i,j] = numpy.random.uniform(low = -alpha, high = alpha)
            symMatrix[j,i] = symMatrix[i,j]
    
    assert(symMatrix[0,0] == 0.0)
    return symMatrix


# def sampleUniformCov(NUMBER_OF_VARIABLES):
#     precisionMatrix = sampleUniformSymmetricMatrix(NUMBER_OF_VARIABLES, 1)
#     precisionMatrix = makePositiveDefinite(precisionMatrix)
#     covMatrix = numpy.linalg.inv(precisionMatrix)
#     corrMatrix = idcHelper.conv2corrMatrix(covMatrix)
#     return corrMatrix

    




# def addUniformNoiseToPrec(fullCovMatrix, uniformNoiseStdOnPrec):
#     noiseMatrix = idcHelper.sampleUniformSymmetricNoise(fullCovMatrix.shape[0], uniformNoiseStdOnPrec)
#     precisionMatrixWithNoise = numpy.linalg.inv(fullCovMatrix)
#     precisionMatrixWithNoise += noiseMatrix
#     eigVals, _ = numpy.linalg.eigh(precisionMatrixWithNoise)
#     if eigVals[0] < 0.001:
#         reg = numpy.abs(eigVals[0]) + 0.001
#         precisionMatrixWithNoise += reg * numpy.eye(precisionMatrixWithNoise.shape[0])
#     
#     fullCovMatrixWithNoise = numpy.linalg.inv(precisionMatrixWithNoise)
#     return fullCovMatrixWithNoise
# 
# 
# def addUniformNoiseToCov(fullCovMatrix, uniformNoiseStdOnCov):
#     noiseMatrix = idcHelper.sampleUniformSymmetricNoise(fullCovMatrix.shape[0], uniformNoiseStdOnCov)
#     covMatrixWithNoise = numpy.copy(fullCovMatrix)
#     covMatrixWithNoise += noiseMatrix
#     eigVals, _ = numpy.linalg.eigh(covMatrixWithNoise)
#     if eigVals[0] < 0.001:
#         reg = numpy.abs(eigVals[0]) + 0.001
#         covMatrixWithNoise += reg * numpy.eye(covMatrixWithNoise.shape[0])
#     return covMatrixWithNoise
# 
# 
# def addWishartNoiseToPrec(fullCovMatrix, wishartNoiseStdOnPrec):
#     noiseMatrix = idcHelper.sampleClusterCovMatrix(fullCovMatrix.shape[0], 0.0)
#     precisionMatrixWithNoise = numpy.linalg.inv(fullCovMatrix)
#     precisionMatrixWithNoise += noiseMatrix
#     eigVals, _ = numpy.linalg.eigh(precisionMatrixWithNoise)
#     if eigVals[0] < 0.001:
#         reg = numpy.abs(eigVals[0]) + 0.001
#         precisionMatrixWithNoise += reg * numpy.eye(precisionMatrixWithNoise.shape[0])
#     
#     fullCovMatrixWithNoise = numpy.linalg.inv(precisionMatrixWithNoise)
#     return fullCovMatrixWithNoise


def getLambdaMin(alpha, fullPrecisionMatrixOnlyBlocks, reducedPrecision):
    X_epsilon = alpha * fullPrecisionMatrixOnlyBlocks + reducedPrecision
    eigVals, _ = numpy.linalg.eigh(X_epsilon)
    lambdaMin = eigVals[0]
    return lambdaMin

def testMatrix(fullCovarianceMatrix, clusterAssignments):
    p = fullCovarianceMatrix.shape[0]
    fullPrecisionMatrix = numpy.linalg.inv(fullCovarianceMatrix)
    
    fullPrecisionMatrixOnlyBlocks = idcHelper.createFullX(p, idcHelper.getBlockCovariance(fullPrecisionMatrix, clusterAssignments))
    reducedPrecision = fullPrecisionMatrix - fullPrecisionMatrixOnlyBlocks
    
    
    alphaMin = 0.0
    alphaMax = 1.0
    alpha = None
    
    for i in range(50):
        assert(alphaMax > alphaMin)
        if (alphaMax - alphaMin) < 0.00001:
            break
        
        
        alpha = (alphaMax + alphaMin) / 2.0
        lambdaMin = getLambdaMin(alpha, fullPrecisionMatrixOnlyBlocks, reducedPrecision)
        
        # print "alphaMin = ", alphaMin
        # print "alphaMax = ", alphaMax
        # print "lambdaMin = ", lambdaMin
    
        if lambdaMin <= 0.0:
            # need to increase alpha
            alphaMin = alpha
        else:
            alphaMax = alpha
    
    alpha += 0.0001
    print("alpha = ", alpha)
    assert(getLambdaMin(alpha, fullPrecisionMatrixOnlyBlocks, reducedPrecision) > 0.0)
    
    return


def createOutlierData(sampleType, NUMBER_OF_VARIABLES, NUMBER_OF_OUTLIER_SAMPLES):
    
    if NUMBER_OF_OUTLIER_SAMPLES == 0:
        return numpy.zeros((0,NUMBER_OF_VARIABLES)) 
    
    if sampleType.startswith("t"):
        print("sample from student t distribution")
        
        # df = 3.0
        # variance = 1.0
        # outlierSamples = scipy.stats.t.rvs(df, loc=0, scale=variance, size=(NUMBER_OF_OUTLIER_SAMPLES, NUMBER_OF_VARIABLES))
        
        noiseMatrix = sampleUniformSymmetricMatrix(NUMBER_OF_VARIABLES, 1)
        noiseMatrix = makePositiveDefinite(noiseMatrix)
        
        # print("before:")
        # print(noiseMatrix)
        # noiseMatrix = normalize(noiseMatrix)
        # print("after:")
        # print(noiseMatrix)
        noiseMatrix = normalize(noiseMatrix)
        
        print("Noise precision matrix = ")
        idcHelper.showMatrix(noiseMatrix)
        
        scaleMatrix = numpy.linalg.inv(noiseMatrix) # scaleMatrix of multivariate t corresponds to the covariance matrix of normal distribution
        # print ("scaleMatrix = ")
        # print(scaleMatrix)
        # scaleMatrix = normalize(scaleMatrix)
        ro.globalenv['scaleMatrix'] = ro.r.matrix(scaleMatrix, nrow = NUMBER_OF_VARIABLES, ncol = NUMBER_OF_VARIABLES)
        
        df = 3
        ro.r('dataVectorsAll = rmvt(' + str(NUMBER_OF_OUTLIER_SAMPLES) + ', sigma = scaleMatrix, df = ' + str(df) + ')')
        outlierSamples = ro.r('as.matrix(dataVectorsAll)')
        ro.r('rm(list = ls())') # for safety: delete all variables in workspace
        return outlierSamples
        
    elif sampleType == "uniform":
        noiseMatrix = sampleUniformSymmetricMatrix(NUMBER_OF_VARIABLES, 1)
        noiseMatrix = makePositiveDefinite(noiseMatrix)
    else:
        assert(sampleType == "invWishart")
        nu0 = NUMBER_OF_VARIABLES + 1
        Sigma0 = numpy.eye(NUMBER_OF_VARIABLES)
        noiseMatrix = scipy.stats.invwishart.rvs(df = nu0, scale = Sigma0, size=1)
    
    modelMean = numpy.zeros(NUMBER_OF_VARIABLES)
    outlierSamples = numpy.random.multivariate_normal(mean = modelMean, cov = noiseMatrix, size = NUMBER_OF_OUTLIER_SAMPLES)
    return outlierSamples

        
def createDataWithOutliers(NUMBER_OF_CLUSTERS, NUMBER_OF_VARIABLES, NUMBER_OF_SAMPLES, sampleType, outlierRatio):
    assert(outlierRatio >= 0.0 and outlierRatio <= 0.1)
    
    # create mean vectors
    modelMeansAppended = numpy.zeros(NUMBER_OF_VARIABLES)
    hiddenDataIds = numpy.zeros(NUMBER_OF_VARIABLES, dtype = numpy.int_)
    
    assert(NUMBER_OF_CLUSTERS >= 2)
    
    trueCovMatrix = numpy.zeros((NUMBER_OF_VARIABLES,NUMBER_OF_VARIABLES))
    
    clusterSizes = numpy.zeros(NUMBER_OF_CLUSTERS, dtype = numpy.int)
        
    # assume balanced clusters
    nrDataPointsClusterPerCluster = int(NUMBER_OF_VARIABLES / NUMBER_OF_CLUSTERS)
    assert(nrDataPointsClusterPerCluster * NUMBER_OF_CLUSTERS == NUMBER_OF_VARIABLES)
    clusterSizes[0:NUMBER_OF_CLUSTERS] = nrDataPointsClusterPerCluster
    
    assert(numpy.sum(clusterSizes) == NUMBER_OF_VARIABLES)
    
    nextClusterStartsAt = 0
    currentClusterId = 0
        
    for i in range(NUMBER_OF_VARIABLES):
        if i == nextClusterStartsAt:
            nrDataPointsClusterPerCluster = clusterSizes[currentClusterId]
            
            # create next cov matrices
            startId = nextClusterStartsAt
            endId = nextClusterStartsAt + nrDataPointsClusterPerCluster
            if sampleType == "uniform":
                covMatrix = sampleUniformSymmetricMatrix(nrDataPointsClusterPerCluster, 1)
                covMatrix = makePositiveDefinite(covMatrix)
            elif sampleType == "invWishart":
                covMatrix = sampleFromInverseWishart(nrDataPointsClusterPerCluster)
            else:
                assert(False)
            trueCovMatrix[startId:endId, startId:endId] = covMatrix
            
            nextClusterStartsAt += nrDataPointsClusterPerCluster
            currentClusterId += 1
            
        hiddenDataIds[i] = currentClusterId
        
    if sampleType == "uniform":
            noiseMatrix = sampleUniformSymmetricMatrix(NUMBER_OF_VARIABLES, 1)
            noiseMatrix = makePositiveDefinite(noiseMatrix)
    else:
        assert(sampleType == "invWishart")
        nu0 = NUMBER_OF_VARIABLES + 1
        Sigma0 = numpy.eye(NUMBER_OF_VARIABLES)
        noiseMatrix = scipy.stats.invwishart.rvs(df = nu0, scale = Sigma0, size=1)
        
    
    NUMBER_OF_TRUE_MODEL_SAMPLES = int((1.0 - outlierRatio) * NUMBER_OF_SAMPLES)
    NUMBER_OF_OUTLIER_SAMPLES = int(outlierRatio * NUMBER_OF_SAMPLES)
    assert(NUMBER_OF_SAMPLES == NUMBER_OF_TRUE_MODEL_SAMPLES + NUMBER_OF_OUTLIER_SAMPLES)
    
    trueSamples = numpy.random.multivariate_normal(mean = modelMeansAppended, cov = trueCovMatrix, size = NUMBER_OF_TRUE_MODEL_SAMPLES)
    outlierSamples = numpy.random.multivariate_normal(mean = modelMeansAppended, cov = noiseMatrix, size = NUMBER_OF_OUTLIER_SAMPLES)
    allSamples = numpy.vstack((trueSamples,outlierSamples))
    assert(allSamples.shape[0] == NUMBER_OF_SAMPLES)
    
    truePrecMatrix = numpy.linalg.inv(trueCovMatrix)
    
    return allSamples, truePrecMatrix


    
def createIndependentDimClusterDataSamples(NUMBER_OF_CLUSTERS, NUMBER_OF_VARIABLES, NUMBER_OF_SAMPLES, clusterSizeDist, sampleType, addNoiseToWhat, noiseType, noiseLevel): 
    assert(addNoiseToWhat == "noNoise" or addNoiseToWhat == "cov" or addNoiseToWhat == "prec")
    
    # create mean vectors
    modelMeansAppended = numpy.zeros(NUMBER_OF_VARIABLES)
    hiddenDataIds = numpy.zeros(NUMBER_OF_VARIABLES, dtype = numpy.int_)
    
    assert(NUMBER_OF_CLUSTERS >= 2)
    
    fullCovMatrix = numpy.zeros((NUMBER_OF_VARIABLES,NUMBER_OF_VARIABLES))
    
    clusterSizes = numpy.zeros(NUMBER_OF_CLUSTERS, dtype = numpy.int)
        
    if clusterSizeDist == "balanced":
        
        nrDataPointsClusterPerCluster = int(NUMBER_OF_VARIABLES / NUMBER_OF_CLUSTERS)
        assert(nrDataPointsClusterPerCluster * NUMBER_OF_CLUSTERS == NUMBER_OF_VARIABLES)
        
        clusterSizes[0:NUMBER_OF_CLUSTERS] = nrDataPointsClusterPerCluster
    
    elif clusterSizeDist == "unbalanced":
        
        assert(NUMBER_OF_VARIABLES == 40 and NUMBER_OF_CLUSTERS == 4)
        clusterSizes[0] = 20
        clusterSizes[1] = 10
        clusterSizes[2] = 5
        clusterSizes[3] = 5
        
    elif clusterSizeDist == "halfLargeHalfSmall":
        
        assert(int(NUMBER_OF_CLUSTERS / 2) * 2 == NUMBER_OF_CLUSTERS)
        
        singleClusterSize = int(NUMBER_OF_VARIABLES / (NUMBER_OF_CLUSTERS + NUMBER_OF_CLUSTERS / 2))
        remainder = NUMBER_OF_VARIABLES - (singleClusterSize * NUMBER_OF_CLUSTERS + singleClusterSize * (NUMBER_OF_CLUSTERS / 2))
        
        clusterSizes[0:(NUMBER_OF_CLUSTERS / 2)] = singleClusterSize * 2
        clusterSizes[(NUMBER_OF_CLUSTERS / 2):NUMBER_OF_CLUSTERS] = singleClusterSize
        clusterSizes[0] += int(remainder)
        
    elif clusterSizeDist == "expDecreasing":
        
        assert(int(NUMBER_OF_CLUSTERS / 2) * 2 == NUMBER_OF_CLUSTERS)
        
        minimalClusterSize = int(NUMBER_OF_VARIABLES / (NUMBER_OF_CLUSTERS * 2))
        clusterSizes[0:NUMBER_OF_CLUSTERS] = minimalClusterSize
        
        remainingMass = NUMBER_OF_VARIABLES - minimalClusterSize * NUMBER_OF_CLUSTERS
        for j in range(NUMBER_OF_CLUSTERS):
            clusterSizes[j] += int(remainingMass / 2)
            remainingMass -= int(remainingMass / 2)
            if remainingMass <= 2:
                break
        
        clusterSizes[0] += int(remainingMass)
        
    else:
        assert(False) 
    
    
    
    assert(numpy.sum(clusterSizes) == NUMBER_OF_VARIABLES)
    
    nextClusterStartsAt = 0
    currentClusterId = 0
        
    for i in range(NUMBER_OF_VARIABLES):
        if i == nextClusterStartsAt:
            nrDataPointsClusterPerCluster = clusterSizes[currentClusterId]
            
            # create next cov matrices
            startId = nextClusterStartsAt
            endId = nextClusterStartsAt + nrDataPointsClusterPerCluster
            if sampleType == "uniformSparseWithin":
                assert(False)
                covMatrix = sampleSparseCov(nrDataPointsClusterPerCluster)
            elif sampleType == "uniform":
                covMatrix = sampleUniformSymmetricMatrix(nrDataPointsClusterPerCluster, 1)
                covMatrix = makePositiveDefinite(covMatrix)
            elif sampleType == "wishart":
                assert(False)
                covMatrix = sampleFromWishart(nrDataPointsClusterPerCluster)
            elif sampleType == "invWishart":
                covMatrix = sampleFromInverseWishart(nrDataPointsClusterPerCluster)
            else:
                assert(False)
            fullCovMatrix[startId:endId, startId:endId] = covMatrix
            
            nextClusterStartsAt += nrDataPointsClusterPerCluster
            currentClusterId += 1
            
        hiddenDataIds[i] = currentClusterId
    
    # fullCovMatrix = idcHelper.conv2corrMatrix(fullCovMatrix)
    
    if addNoiseToWhat != "noNoise":
        assert(noiseLevel >= 0.001 and noiseLevel <= 0.2)
        
        if noiseType == "uniform":
            noiseMatrix = sampleUniformSymmetricMatrix(NUMBER_OF_VARIABLES, 1)
            noiseMatrix = makePositiveDefinite(noiseMatrix)
        else:
            assert(noiseType == "invWishart")
            nu0 = NUMBER_OF_VARIABLES + 1
            Sigma0 = numpy.eye(NUMBER_OF_VARIABLES)
            noiseMatrix = scipy.stats.invwishart.rvs(df = nu0, scale = Sigma0, size=1)
            
        if addNoiseToWhat == "prec":
            fullCovMatrix = numpy.linalg.inv(numpy.linalg.inv(fullCovMatrix) + noiseLevel * numpy.linalg.inv(noiseMatrix))
        elif addNoiseToWhat == "cov":
            fullCovMatrix += noiseLevel * noiseMatrix
        else:
            print("no noise added")
    
    
    # testMatrix(fullCovMatrix, hiddenDataIds)
    # assert(False)
    
    allDimSamples = numpy.random.multivariate_normal(mean = modelMeansAppended, cov = fullCovMatrix, size = NUMBER_OF_SAMPLES)
        
    for i in range(NUMBER_OF_VARIABLES):
        assert(hiddenDataIds[i] >= 1 and hiddenDataIds[i] <= NUMBER_OF_CLUSTERS)
    
    precisionMatrix = numpy.linalg.inv(fullCovMatrix)
    
    print("finished creation of data: gaussian cluster data with conditionally independent dimensions.")
    return allDimSamples, hiddenDataIds, NUMBER_OF_CLUSTERS, fullCovMatrix, precisionMatrix



def getSimulatedDataFilename(sampleType, addNoiseToWhat, noiseType, noiseLevel):
    
    if addNoiseToWhat == "noNoise":
        return sampleType
    else:
        assert(noiseLevel >= 0.01) # otherwise we cannot save in percent
        assert(addNoiseToWhat == "prec" or addNoiseToWhat == "cov")
        return sampleType + "_" + addNoiseToWhat + "_" + str(int(noiseLevel * 100)) + "%" + noiseType
    

def createHomogeneousClusters(NUMBER_OF_CLUSTERS, NUMBER_OF_VARIABLES, NUMBER_OF_SAMPLES, corrValue): 
    
    # create mean vectors
    modelMeansAppended = numpy.zeros(NUMBER_OF_VARIABLES)
    hiddenDataIds = numpy.zeros(NUMBER_OF_VARIABLES, dtype = numpy.int_)
    
    assert(NUMBER_OF_CLUSTERS >= 2)
    
    currentClusterId = 1
    nrDataPointsClusterPerCluster = int(NUMBER_OF_VARIABLES / NUMBER_OF_CLUSTERS)
    assert(nrDataPointsClusterPerCluster * NUMBER_OF_CLUSTERS == NUMBER_OF_VARIABLES)
    
    fullCovMatrix = numpy.zeros((NUMBER_OF_VARIABLES,NUMBER_OF_VARIABLES))
    
    
    # add corr matrix of first cluster
    corrMatrix = idcHelper.conv2corrMatrix(createHomogenousCorr(nrDataPointsClusterPerCluster, corrValue))
    fullCovMatrix[0:nrDataPointsClusterPerCluster, 0:nrDataPointsClusterPerCluster] = corrMatrix
            
    # use uniform distribution for clusters
    for i in range(NUMBER_OF_VARIABLES):
        if i >= currentClusterId * nrDataPointsClusterPerCluster:
            currentClusterId += 1
            
            # create next cov matrices
            startId = (currentClusterId - 1) * nrDataPointsClusterPerCluster
            endId = currentClusterId * nrDataPointsClusterPerCluster
            corrMatrix = idcHelper.conv2corrMatrix(createHomogenousCorr(nrDataPointsClusterPerCluster, corrValue))
            fullCovMatrix[startId:endId, startId:endId] = corrMatrix
            
        hiddenDataIds[i] = currentClusterId
    

    allDimSamples = numpy.random.multivariate_normal(mean = modelMeansAppended, cov = fullCovMatrix, size = NUMBER_OF_SAMPLES)
        
    for i in range(NUMBER_OF_VARIABLES):
        assert(hiddenDataIds[i] >= 1 and hiddenDataIds[i] <= NUMBER_OF_CLUSTERS)
    
    precisionMatrix = numpy.linalg.inv(fullCovMatrix)
    
    print("finished creation of data: gaussian cluster data with conditionally independent dimensions.")
    return allDimSamples, hiddenDataIds, NUMBER_OF_CLUSTERS, fullCovMatrix, precisionMatrix



def createHomogenousCorr(NUMBER_OF_VARIABLES, value):
    precisionMatrix = numpy.ones((NUMBER_OF_VARIABLES,NUMBER_OF_VARIABLES))
    for i in range(NUMBER_OF_VARIABLES):
        for j in range(i+1,NUMBER_OF_VARIABLES):
            precisionMatrix[i,j] = value
            precisionMatrix[j,i] = precisionMatrix[i,j]
         
    # precisionMatrix = makePositiveDefinite(precisionMatrix)
    # covMatrix = numpy.linalg.inv(precisionMatrix)
    # corrMatrix = idcHelper.conv2corrMatrix(covMatrix)
    return precisionMatrix


    

