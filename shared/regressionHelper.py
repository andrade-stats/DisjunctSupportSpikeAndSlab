import numpy

# checked
def getSubMatriciesOfDataMatrix(X, cIndicies):
    sIndicies = numpy.arange(X.shape[1])
    sIndicies = numpy.delete(sIndicies, cIndicies) 
    Xs = X[:,sIndicies]
    Xc = X[:,cIndicies]
    return Xs, Xc

# checked
def getBetaEstimateML(Xs, y):
    # ridgeWeight = 0.001
    # p = Xs.shape[1]
    # invXTX = numpy.linalg.inv(numpy.matmul(Xs.transpose(), Xs) + ridgeWeight * numpy.eye(p))
    
    invXTX = numpy.linalg.inv(numpy.matmul(Xs.transpose(), Xs))
    XTy = numpy.matmul(Xs.transpose(), y)
    return numpy.matmul(invXTX, XTy)

# checked
def getLinearRegEBIC(X, y, zeroIndicies, gamma = 0.0):
    assert(X.shape[0] == y.shape[0])
    assert(zeroIndicies.shape[0] <= X.shape[1])
    n = X.shape[0]
    p = X.shape[1]
    nrOfFreeParameters = X.shape[1] - zeroIndicies.shape[0] + 1 # added 1 since the variance of the normal distribution form the noise epsilon is also a free parameter
    
    if zeroIndicies.shape[0] > 0:
        Xs, Xc = getSubMatriciesOfDataMatrix(X, zeroIndicies)
    else:
        Xs = X
    
    betaMLs = getBetaEstimateML(Xs, y)
    squaredResidualSum = numpy.sum(numpy.square(y - numpy.matmul(Xs, betaMLs))) # also called RSS
    minus2logLikelihood = n + n * numpy.log(2.0 * numpy.pi) + n * numpy.log(squaredResidualSum / n) 
    # bic = minus2logLikelihood +  nrOfFreeParameters * numpy.log(n)  
    EBICcriteria = minus2logLikelihood + nrOfFreeParameters * numpy.log(n) + 2.0 * nrOfFreeParameters * gamma * numpy.log(p)
    return EBICcriteria

# checked
def getLinearRegAIC(X, y, zeroIndicies):
    assert(X.shape[0] == y.shape[0])
    assert(zeroIndicies.shape[0] <= X.shape[1])
    n = X.shape[0]
    p = X.shape[1]
    nrOfFreeParameters = X.shape[1] - zeroIndicies.shape[0] + 1 # added 1 since the variance of the normal distribution form the noise epsilon is also a free parameter
    
    if zeroIndicies.shape[0] > 0:
        Xs, Xc = getSubMatriciesOfDataMatrix(X, zeroIndicies)
    else:
        Xs = X
    
    betaMLs = getBetaEstimateML(Xs, y)
    squaredResidualSum = numpy.sum(numpy.square(y - numpy.matmul(Xs, betaMLs))) # also called RSS
    minus2logLikelihood = n + n * numpy.log(2.0 * numpy.pi) + n * numpy.log(squaredResidualSum / n) 
    AICcriteria = minus2logLikelihood + 2.0 * nrOfFreeParameters 
    return AICcriteria
