import numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def simpleOneDimNormalizationConstantEstimate(unnormalizedDensityFunction, samples):
    sortedSamples = numpy.sort(samples)
    
    unnormalizedDensityValues = unnormalizedDensityFunction(sortedSamples)
    normalizationEstimate = 0.0
    for i in range(1, len(samples)):
        width = sortedSamples[i] - sortedSamples[i-1]
        height = (unnormalizedDensityValues[i] + unnormalizedDensityValues[i-1]) / 2.0
        assert(width > 0 and height > 0)
        normalizationEstimate += width * height
    
    return normalizationEstimate


def plotHistogramWithEstimateNormalizedDensity(unnormalizedDensityFunction, samples, numberOfBins = 30):
    
    normalizationConstantEstimate = simpleOneDimNormalizationConstantEstimate(unnormalizedDensityFunction, samples)
    print("normalizationConstantEstimate = ", normalizationConstantEstimate)

    counts, bins, _ = plt.hist(samples, numberOfBins, density=True)
    plt.plot(bins, unnormalizedDensityFunction(bins) / normalizationConstantEstimate, linewidth=2, color='r')
    plt.savefig("/Users/danielandrade/workspace/RobustExtensions/plots/histogramWithDensity.pdf")
    return