import numpy
import shared.idcHelper as idcHelper
import matplotlib.pyplot as plt

# def showHeatMap(M):
#     
#     fig = matplotlib.pyplot.figure()
#     ax1 = fig.add_subplot(111)
#     cmap = matplotlib.cm.get_cmap('Reds', 30)
#     cax = ax1.imshow(M, interpolation="nearest", cmap=cmap)
#     ax1.grid(True)
#     # plt.title('Abalone Feature Correlation')
#     # labels=['Sex','Length','Diam','Height','Whole','Shucked','Viscera','Shell','Rings',]
#     # ax1.set_xticklabels(labels,fontsize=6)
#     # ax1.set_yticklabels(labels,fontsize=6)
#     # Add colorbar, make sure to specify tick locations to match desired ticklabels
#     fig.colorbar(cax) # , ticks=[.75,.8,.85,.90,.95,1])
#     matplotlib.pyplot.show()

def showCovPrec(covM):
    print("ANALYZE COVARIANCE MATRIX:")
    idcHelper.showMatrix(covM)
    showHeatMapNew(numpy.asarray(covM))
    
    precM = numpy.linalg.inv(covM)
    print("ANALYZE PRECISION MATRIX:")
    idcHelper.showMatrix(precM)
    showHeatMapNew(numpy.asarray(precM))
    return
    
# tested
def showHeatMapNew(Morig):
    
    M = numpy.copy(Morig)
    
    # set values very close to 0 to 0:
    ZERO_APPROX = 0.0001
    M[numpy.logical_and(M <= ZERO_APPROX, M >= -ZERO_APPROX)] = 0.0
    
    # this is the normal situation. Special matrices like matrices with constant value are not changed
    if numpy.min(M) < 0 and numpy.max(M) > 0:
    
        NUMBER_OF_LEVELS = 10
        
        assert(numpy.min(M) < 0 and numpy.max(M) > 0)
        thresholdOnBothSides = min(- numpy.min(M), numpy.max(M)) # is used in order to ensure symmetry of the colorbar
        
        print("thresholdOnBothSides = ")
        print(thresholdOnBothSides)
        
        M[M <= -thresholdOnBothSides] = -thresholdOnBothSides
        M[M >= thresholdOnBothSides] = thresholdOnBothSides
        
        for i in range(NUMBER_OF_LEVELS):
            stepSize = thresholdOnBothSides / NUMBER_OF_LEVELS
            lowerBound = i * stepSize
            upperBound = (i+1) * stepSize
            M[numpy.logical_and(M <= upperBound, M > lowerBound)] = upperBound
            M[numpy.logical_and(M < -lowerBound, M >= -upperBound)] = - upperBound
        
        
    CS = plt.pcolor(M, cmap=plt.cm.seismic)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()


def test():
    M = numpy.asarray([[1.0, 0.08, -0.6, 0.3], [-0.08, 1.0, 0.5, 0.2], [0.6, 0.5, 1.0, 0.9], [0.3, -0.2, 0.9, -0.00001]])
    
#     M = numpy.random.random((7, 7))
#     M[0,0] = 0.0
#     M[2,3] = -5
#     M[4,4] = -0.1
#     M[5,5] = 0.1
#     M[5,6] = 6.0
    
    # print "M = "
    # print M
    # showHeatMapNew(M)
    
    M = numpy.asarray([[1.0, 0.08, -0.6, 0.3], [-0.08, 1.0, 0.5, 0.2], [0.6, 0.5, 1.0, 0.9]])
    # M = numpy.asmatrix(M)
    print("M = ")
    print(M)
    vec = numpy.asmatrix(numpy.asarray([1, 2, 3, 4]))
    ones = numpy.asmatrix(numpy.ones(3))
    
    result = numpy.multiply(ones.transpose() * vec, M)
    print("result = ")
    print(result)
    
# test()
