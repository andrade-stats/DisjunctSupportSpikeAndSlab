
import numpy

# write.table(XDLBCL, file = "/Users/danielandrade/workspace/StanTest/datasets/test.txt", row.names = FALSE, col.names = FALSE)

def getLineCount(filename):
    numLines = sum(1 for line in open(filename, "r"))
    return numLines

def getColumnCountR(filename):
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            return len(line.split(" "))


# tested
# assumes data was saved using
# write.table(golub, file = "/Users/danielandrade/workspace/StanTest/datasets/golub_plain.txt", row.names = FALSE, col.names = FALSE)
def loadMatrixFromR(filename):
    NUMBER_OF_OBSERVATIONS = getLineCount(filename)
    NUMBER_OF_VARIABLES = getColumnCountR(filename)
    
    dataSamples = numpy.zeros((NUMBER_OF_OBSERVATIONS, NUMBER_OF_VARIABLES))
    
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            for j, elem in enumerate(line.split(" ")):
                dataSamples[i, j] = float(elem)
    
    return dataSamples


# tested
# assumes data was saved using
# write.table(golub.cl, file = "/Users/danielandrade/workspace/StanTest/datasets/golub_labels.txt", row.names = FALSE, col.names = FALSE)
def loadClassLabelsFromR(filename):
    NUMBER_OF_VARIABLES = getLineCount(filename)
    correctLabels = numpy.zeros(NUMBER_OF_VARIABLES)
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            correctLabels[i] = float(line)
            assert(correctLabels[i] >= 0)
    
    return correctLabels.astype(int)


# dataSamples = loadMatrixFromR("/Users/danielandrade/workspace/StanTest/datasets/test.txt")
# print dataSamples.shape