
import numpy
import re
import shared.ioHelper as ioHelper
import scipy.io
import csv

# 1: NZD   
# 2: AUD
# 3: JPY 
# 4: SEK  
# 5: GBP   
# 6: ESP   
# 7: BEF   
# 8: FRF   
# 9: CHF 
# 10: NLG   
# 11: DEM
# shortcuts are explained in Figure 1, "Simulation of hyper-inverse Wishart distributions in graphical models"
# data downloaded from http://www2.stat.duke.edu/research/software/west/hiwsim.html
def loadExchangeRateDataSmall(pathprefix):
    d = scipy.io.loadmat(pathprefix + "datasets/HIWcodeData.mat")
    
    # no ground truth, therefore cluster labels are set to roughly meet the structure in Figure 1, "Simulation of hyper-inverse Wishart distributions in graphical models"
    clusterLabels = numpy.ones(11, dtype = numpy.int)
    
    clusterLabels[0] = 2
    clusterLabels[1] = 2
    clusterLabels[3] = 3
    clusterLabels[4] = 4
    
    # clusterLabels *= 1
    # clusterLabels[1, 1, 2, 2 , ]
    return d["X"], clusterLabels, 4



# "demonfx" from LaplacesDemon 
def loadExchangeRateDataLarge(pathprefix):

    variableNames = None
    closingIndicies = numpy.arange(start = 0, stop = 39, step = 3)
    filename = "/Users/danielandrade/workspace/StanTest/datasets/demonfx.csv"
    
    numberOfSamples = 1301
    numberOfVariables = 13
    
    dataVectors = numpy.zeros(shape = (numberOfSamples, numberOfVariables))
    
    with open(filename,'r') as f:
        for lineNr, elemsInLine in enumerate(csv.reader(f)):
            assert(len(elemsInLine) == 40)
            allRelElems = elemsInLine[1:40]
            
            if lineNr == 0:
                # get all variable names
                allRelElems = numpy.asarray(allRelElems)
                variableNames = allRelElems[closingIndicies]
                
            else:
                allRelElems = [float(elem) for elem in allRelElems]
                allRelElems = numpy.asarray(allRelElems)
                allClosingValues = allRelElems[closingIndicies]
                dataVectors[lineNr-1] = allClosingValues
                
            assert(lineNr <= 1301)
    
    print("variableNames = ")
    print(variableNames)
    
    # dummy cluster labels
    clusterLabels = numpy.ones(numberOfVariables, dtype = numpy.int)
    return dataVectors, clusterLabels, 1


# WARNING: should better be used as n = 38, and p = 3051, otherwise multivariate gaussian justification not clear
# data from R package "multtest"
# "gene expression values from 3051 genes taken from 38 Leukemia patients. Twenty seven patients are diagnosed as acute lymphoblastic leukemia (ALL) and eleven as acute myeloid leukemia (AML)"
# see http://www.bioconductor.org/packages/release/bioc/manuals/multtest/man/multtest.pdf
def loadGolubData_forVariableClustering(pathprefix):
    dataSamples = ioHelper.loadMatrixFromR(pathprefix + "datasets/golub_plain.txt")
    labels = ioHelper.loadClassLabelsFromR(pathprefix + "datasets/golub_labels.txt")
    labels = labels + 1
    numberOfClusters = 2
    assert(dataSamples.shape[1] == labels.shape[0])
    assert(numpy.min(labels) == 1)
    assert(numberOfClusters == numpy.max(labels))
    return dataSamples, labels, numberOfClusters

# data from R package "fabiaData"
# see https://www.bioconductor.org/packages/3.7/bioc/vignettes/fabia/inst/doc/fabia.pdf
# and also http://www.bioconductor.org/packages/release/data/experiment/manuals/fabiaData/man/fabiaData.pdf
def loadDLBCL_forVariableClustering(pathprefix):
    dataSamples = ioHelper.loadMatrixFromR(pathprefix + "datasets/DLBCL_plain.txt")
    labels = ioHelper.loadClassLabelsFromR(pathprefix + "datasets/DLBCL_labels.txt")
    numberOfClusters = 3
    assert(dataSamples.shape[1] == labels.shape[0])
    assert(numberOfClusters == numpy.max(labels))
    assert(numpy.min(labels) == 1)
    return dataSamples, labels, numberOfClusters

# data from R package "fabiaData"
# see https://www.bioconductor.org/packages/3.7/bioc/vignettes/fabia/inst/doc/fabia.pdf
# and also http://www.bioconductor.org/packages/release/data/experiment/manuals/fabiaData/man/fabiaData.pdf
def loadMulti_forVariableClustering(pathprefix):
    dataSamples = ioHelper.loadMatrixFromR(pathprefix + "datasets/Multi_plain.txt")
    labels = ioHelper.loadClassLabelsFromR(pathprefix + "datasets/Multi_labels.txt")
    numberOfClusters = 4
    assert(dataSamples.shape[1] == labels.shape[0])
    assert(numpy.min(labels) == 1)
    assert(numberOfClusters == numpy.max(labels))
    return dataSamples, labels, numberOfClusters

# data from R package "fabiaData"
# see https://www.bioconductor.org/packages/3.7/bioc/vignettes/fabia/inst/doc/fabia.pdf
# and also http://www.bioconductor.org/packages/release/data/experiment/manuals/fabiaData/man/fabiaData.pdf
def loadBreast_forVariableClustering(pathprefix):
    dataSamples = ioHelper.loadMatrixFromR(pathprefix + "datasets/Breast_plain.txt")
    labels = ioHelper.loadClassLabelsFromR(pathprefix + "datasets/Breast_labels.txt")
    numberOfClusters = 3
    assert(dataSamples.shape[1] == labels.shape[0])
    assert(numpy.min(labels) == 1)
    assert(numberOfClusters == numpy.max(labels))
    return dataSamples, labels, numberOfClusters

# data from "Flight Data For Tail 687"
# https://c3.nasa.gov/dashlink/resources/664/
def loadAviationData(pathprefix, nrVariables, files):
    
    if nrVariables == 15:
        dataVectors = numpy.load(pathprefix + "datasets/aviationData_15vars_allFlights.npy")
    elif nrVariables == 77:
        if files == 1:
            dataVectors = numpy.load(pathprefix + "datasets/aviationData_77vars_allFlights_oneFile.npy")
        elif files == 2:
            dataVectors = numpy.load(pathprefix + "datasets/aviationData_77vars_allFlights_twoFiles.npy")
        elif files == 10:
            dataVectors = numpy.load(pathprefix + "datasets/aviationData_77vars_allFlights_allFiles.npy")
        else:
            assert(False)
    else:
        assert(False)
        
    numberOfVariables = dataVectors.shape[1]
    clusterLabels = numpy.ones(numberOfVariables, dtype = numpy.int)
    return dataVectors, clusterLabels, 1

def getIdsFromString(idsStr):
    idsStrSplit = idsStr.split(" ")
    
    idsNumpy = numpy.zeros(len(idsStrSplit), dtype = numpy.int64)
    
    for i in range(len(idsStrSplit)):
        idsNumpy[i] = int(idsStrSplit[i])
    
    return idsNumpy

def showAviationClusteringResult(pathprefix, clusteringResult):
    if clusteringResult is None:
        return
    
    if clusteringResult.shape[0] == 15:
        allRelevantKeysDescriptions = numpy.load(pathprefix + "datasets/relevantKeysAviationDataDescriptions.npy")
    elif clusteringResult.shape[0] == 16:
        allRelevantKeysDescriptions = numpy.load(pathprefix + "/datasets/relevantDescriptionsAviationData_allOver500.npy")
        selectedVarIdsStr = "54 6 56 61 2 55 18 19 58 8 4 3 1 63 9 64"
        allRelevantKeysDescriptions = allRelevantKeysDescriptions[getIdsFromString(selectedVarIdsStr)]
        assert(allRelevantKeysDescriptions.shape[0] == 16)
    elif clusteringResult.shape[0] == 77:
        allRelevantKeysDescriptions = numpy.load(pathprefix + "/datasets/relevantDescriptionsAviationData_allOver500.npy")
    elif clusteringResult.shape[0] == 57:
        allRelevantKeysDescriptions = numpy.load(pathprefix + "/datasets/relevantDescriptionsAviationData_allOver500.npy")
        proposedClusteringFull = "1 9 9 1 1 1 1 1 2 1 1 13 4 10 1 1 1 1 1 1 1 12 1 13 11 1 5 8 1 1 13 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 4 1 1 7 6 1 1 13 13 13 1 1 1 1 3 13 1"
        clusterAssignments = getIdsFromString(proposedClusteringFull)
        allRelevantKeysDescriptions = allRelevantKeysDescriptions[clusterAssignments == 1]
    elif clusteringResult.shape[0] == 75:
        allRelevantKeysDescriptions = numpy.load(pathprefix + "/datasets/relevantDescriptionsAviationData_allOver500.npy")
        BICClustering = "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1"
        clusterAssignments = getIdsFromString(BICClustering)
        allRelevantKeysDescriptions = allRelevantKeysDescriptions[clusterAssignments == 1]
    else:
        assert(False)
    
    assert(allRelevantKeysDescriptions.shape[0] == clusteringResult.shape[0])
    numberOfClusters = numpy.max(clusteringResult)
    for clusterId in range(1, numberOfClusters+1, 1):
        ids = numpy.where(clusteringResult == clusterId)[0]
        print("\\footnotesize Cluster " + str(clusterId) + " & \\footnotesize " + ", ".join(allRelevantKeysDescriptions[ids]))
        print("\\midrule")
        assert(len(ids) >= 1)
    
    return 


# checked
# return data matrix with format (number of rows, number of columns) = (number of samples, number of variables)
# in order to normalize each variable use "statHelper.normalizeData(dataVectors)"
# from "Feature-inclusion Stochastic Search for Gaussian Graphical Models" (supplement)
# and also used for example in "Group Sparse Priors for Covariance Estimation"
def loadMutualFundData_forVariableClustering(pathprefix):
    
    numberOfSamples = 86
    numberOfVariables = 59
    
    hiddenVarIds = numpy.zeros(numberOfVariables, dtype = numpy.int_)
    dataVectors = numpy.zeros(shape = (numberOfSamples, numberOfVariables))
    numberOfClusters = 4
    
    filename = pathprefix + "datasets/mutual_fund_data.txt"
    for sampleId, line in enumerate(open(filename, "r")):
        line = line.strip()
        allNums = line.split(" ")
        assert(len(allNums) == numberOfVariables)
        for varId, num in enumerate(allNums):
            dataVectors[sampleId,varId] = float(num)
    
    USbondFunds = 13
    USstockFunds = 30
    balancedFunds = 7
    internationalStockFunds = 9
    
    # assign labels to variables
    
    for varId in range(0, USbondFunds, 1):
        hiddenVarIds[varId] = 1
    
    for varId in range(USbondFunds, USbondFunds + USstockFunds, 1):
        hiddenVarIds[varId] = 2
    
    for varId in range(USbondFunds + USstockFunds, balancedFunds + USbondFunds + USstockFunds, 1):
        hiddenVarIds[varId] = 3
        
    for varId in range(balancedFunds + USbondFunds + USstockFunds, internationalStockFunds + balancedFunds + USbondFunds + USstockFunds, 1):
        hiddenVarIds[varId] = 4
    
    print("loaded data successfully")
    return dataVectors, hiddenVarIds, numberOfClusters
    
# checked
# return data matrix with format (number of rows, number of columns) = (number of samples, number of variables)
# in order to normalize each variable use "statHelper.normalizeData(dataVectors)"
# is from the huge package in R
# is used for example in "Adaptive Variable Clustering in Gaussian Graphical Models"
# they achieve: the mean and the standard deviation of the Rand Index are 0.89 and 0.007.
def loadStockDataSP500_forVariableClustering(pathprefix):
    
    filename = pathprefix + "datasets/stockdataSP500.txt"
    filenameLables = pathprefix + "datasets/stockdataSP500_legend.txt"
    
    numberOfSamples = 1258
    numberOfVariables = 452
    
    dataVectors = numpy.zeros(shape = (numberOfSamples, numberOfVariables))
    
    for sampleId, line in enumerate(open(filename, "r")):
        line = line.strip()
        allNums = line.split(" ")
        # print(len(allNums))
        assert(len(allNums) == numberOfVariables)
        for varId, num in enumerate(allNums):
            dataVectors[sampleId,varId] = float(num)
    
    
    hiddenVarIds = numpy.zeros(numberOfVariables, dtype = numpy.int_)
    
    labelsToClusterId = {}
    for varId, line in enumerate(open(filenameLables, "r")):
        line = line.strip()
        label = (line.split("\t")[1]).strip()
        label = re.match("\"(.*)\"", label).group(1) # remove quotation marks
        if label not in labelsToClusterId.keys():
            labelsToClusterId[label] = len(labelsToClusterId) + 1
        
        hiddenVarIds[varId] = labelsToClusterId[label]
        
    numberOfClusters = len(labelsToClusterId)
    
    # print "all labels = "
    # print(labelsToClusterId)
    
    print("loaded data successfully")
    return dataVectors, hiddenVarIds, numberOfClusters        



def loadStockDataSP500_forVariableClusteringSubset(pathprefix):
    
    filename = pathprefix + "datasets/stockdataSP500.txt"
    filenameLables = pathprefix + "datasets/stockdataSP500_legend.txt"
    
    numberOfSamples = 1258
    numberOfVariables = 452
    
    dataVectors = numpy.zeros(shape = (numberOfSamples, numberOfVariables))
    
    for sampleId, line in enumerate(open(filename, "r")):
        line = line.strip()
        allNums = line.split(" ")
        # print(len(allNums))
        assert(len(allNums) == numberOfVariables)
        for varId, num in enumerate(allNums):
            dataVectors[sampleId,varId] = float(num)
    
    
    # hiddenVarIds = numpy.zeros(numberOfVariables, dtype = numpy.int_)
    hiddenVarIds = []
    selectedVariables = []
    
    # labelsToClusterId = {}
    for varId, line in enumerate(open(filenameLables, "r")):
        # print "varId = ", varId
        # assert(False)
        line = line.strip()
        label = (line.split("\t")[1]).strip()
        label = re.match("\"(.*)\"", label).group(1) # remove quotation marks
        # if label not in labelsToClusterId.keys():
        #    labelsToClusterId[label] = len(labelsToClusterId) + 1
        
        if label == "Utilities":
            selectedVariables.append(varId)
            hiddenVarIds.append(1)
        elif label == "Information Technology":
            selectedVariables.append(varId)
            hiddenVarIds.append(2)
        
    
    hiddenVarIds = numpy.asarray(hiddenVarIds, dtype = numpy.int)
    numberOfClusters = numpy.max(hiddenVarIds)
    
    print("selectedVariables = ")
    print(selectedVariables)
    print("hiddenVarIds = ")
    print(hiddenVarIds)
    print("numberOfClusters = ", numberOfClusters)
    
    dataVectors = dataVectors[:,selectedVariables]
    
    # assert(False)
    
    print("loaded data successfully")
    return dataVectors, hiddenVarIds, numberOfClusters      

# checked
# return data matrix with format (number of rows, number of columns) = (number of samples, number of variables)
# in order to normalize each variable use "statHelper.normalizeData(dataVectors)"
# used for example in "The cluster graphical lasso for improved estimation of Gaussian graphical models", 2015
def loadArabidopsisThalianaData_forVariableClustering(pathprefix):
    
    numberOfSamples = 118
    numberOfVariables = 39
    
    hiddenVarIds = numpy.zeros(numberOfVariables, dtype = numpy.int_)
    dataVectors = numpy.zeros(shape = (numberOfSamples, numberOfVariables))
    numberOfClusters = 2
    
    filename = pathprefix + "datasets/arabidopsis_thaliana_data.txt"
    labelsToClusterId = {}
    labelsToClusterId["Mevalonatepathway"] = 1
    labelsToClusterId["Non-Mevalonatepathway"] = 2
    nrOfDescriptionCols = 6
    for varId, line in enumerate(open(filename, "r")):
        line = line.strip()
        allParts = line.split(" ")
        label = allParts[0]
        assert(len(allParts) == numberOfSamples + nrOfDescriptionCols)
        assert(label in labelsToClusterId.keys())
        for sampleId, num in enumerate(allParts[nrOfDescriptionCols:(numberOfSamples + nrOfDescriptionCols)]):
            dataVectors[sampleId, varId] = float(num)
        
        hiddenVarIds[varId] = labelsToClusterId[label]
    
    print("loaded data successfully")
    return dataVectors, hiddenVarIds, numberOfClusters


# def loadOlivettiFaces_forVariableClustering(pathprefix):
#     numberOfClusters = 2
#     numberOfPictures = numberOfClusters * 10
#     
#     dataVectors = numpy.load(pathprefix + "datasets/olivettifaces_plain.npy")
#     
#     dataVectors = dataVectors[0:numberOfPictures, :]
#     dataVectors = dataVectors.transpose()
#     
#     hiddenVarIds = numpy.load(pathprefix + "datasets/olivettifaces_labels.npy")
#     hiddenVarIds = hiddenVarIds[0:numberOfPictures]
#     hiddenVarIds += 1
#     
#     numberOfClusters = numpy.max(hiddenVarIds)
#     return dataVectors, hiddenVarIds, numberOfClusters

def loadOlivettiFaces_forVariableClustering(pathprefix):
    numberOfClusters = 10
    numberOfPictures = numberOfClusters * 10
    
    dataVectors = numpy.load(pathprefix + "datasets/olivettifaces_plain.npy")
    print("dataVectors.shape = ", dataVectors.shape)
    
    
    dataVectors = dataVectors[0:numberOfPictures, :]
    dataVectors = dataVectors.transpose()
    
    hiddenVarIds = numpy.load(pathprefix + "datasets/olivettifaces_labels.npy")
    hiddenVarIds = hiddenVarIds[0:numberOfPictures]
    hiddenVarIds += 1
    
    # print "hiddenVarIds = ", hiddenVarIds
    # assert(False)
    
    numberOfClusters = numpy.max(hiddenVarIds)
    return dataVectors, hiddenVarIds, numberOfClusters


# from 109th Senate Roll Data at 
# http://www.voteview.com/senate109.htm
# encoding used here:
# 0 = no voting
# 1 = voting yes
# -1 = voting no
def loadCongressVotes_forVariableClustering(pathprefix):
    numberOfClusters = 3
    d = scipy.io.loadmat(pathprefix + "datasets/senate109_.mat")
    dataVectors = d["xVote"]
    dataVectors = dataVectors.transpose()
    hiddenVarIds = d["xPartyMask"][:,0]
    hiddenVarIds[hiddenVarIds == 200] = 2 # 200  Republican
    hiddenVarIds[hiddenVarIds == 100] = 1 # 100  Democrat
    hiddenVarIds[hiddenVarIds == 328] = 3 # 328  Independent
    # print "votes of first guy:"
    # print d["names"][-1]
    # print dataVectors[:,1]
    nameList = loadPoliticianNames(pathprefix)
    assert(len(nameList) == hiddenVarIds.shape[0]) 
    
    return dataVectors, hiddenVarIds, numberOfClusters, nameList

def showClusteredNamesCongreeVotes(clusterAssignments, hiddenVarIds, nameList):
    assert(clusterAssignments.shape[0] == hiddenVarIds.shape[0])
    assert(numpy.min(clusterAssignments) == 1)
    assert(numpy.min(hiddenVarIds) == 1)
    assert(clusterAssignments.shape[0] == len(nameList))
    partyNameMap = {}
    partyNameMap[1] = "D"
    partyNameMap[2] = "R"
    partyNameMap[3] = "I"
    for z in range(1, numpy.max(clusterAssignments) + 1):
        print("********************")
        print("Cluster ", z)
        for i in range(clusterAssignments.shape[0]):
            if clusterAssignments[i] == z:
                print(nameList[i] + " (" + str(partyNameMap[hiddenVarIds[i]]) + ") \\\\")
    return

def loadPoliticianNames(pathprefix):
    filename = pathprefix + "datasets/sen109kh.ord"
    nameList = []
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            matchObj = re.match(r'(\d*)\s+\d(\w+)\s+\d+(\w+)\s+(\d+)', line)
            state = matchObj.group(2)  
            name = matchObj.group(3)
            # fullName = name + " (" + state + ")"
            nameList.append(name.strip())
            
    return nameList
       
# line = "1091563368 0WYOMING 20001THOMAS     911116661111666666166666661616"
# line = "1099991099 0USA     20000BUSH       911999"
# matchObj = re.match(r'(\d*)\s+\d(\w+)\s+\d+(\w+)\s+(\d+)', line)
# state = matchObj.group(2)  
# name = matchObj.group(3)
# fullName = name + " (" + state + ")"
# print fullName

# pathprefix = "../../"
# loadOlivettiFaces_forVariableClustering(pathprefix)
# import sklearn.datasets
# d = sklearn.datasets.fetch_olivetti_faces()
# print d.data
# numpy.save("../../olivettifaces_plain", d.data)
# numpy.save("../../olivettifaces_labels", d.target)

# pathprefix = "../../"
# d = scipy.io.loadmat(pathprefix + "datasets/senate109_.mat")
# print d.keys()
# print d["xVote"].shape
# # print d["names"].shape
# print d["xPartyMask"].shape
# # print d["names"][0]
# # print d["xVote"][101]
# hiddenVarIds = d["xPartyMask"][:,0]
# hiddenVarIds[hiddenVarIds == 200] = 2
# hiddenVarIds[hiddenVarIds == 100] = 1
# hiddenVarIds[hiddenVarIds == 328] = 3


                             
# "geneExpression" from r pckage BDgraph 
def loadGeneExpression(pathprefix):

    variableNames = None
    filename = pathprefix + "datasets/geneExpression.csv"
    
    numberOfSamples = 60
    numberOfVariables = 100
    
    dataVectors = numpy.zeros(shape = (numberOfSamples, numberOfVariables))
    
    with open(filename,'r') as f:
        for lineNr, elemsInLine in enumerate(csv.reader(f)):
            assert(len(elemsInLine) == numberOfVariables + 1)
            allRelElems = elemsInLine[1:numberOfVariables + 1]
            
            if lineNr == 0:
                # get all variable names
                variableNames = numpy.asarray(allRelElems)
                
            else:
                allRelElems = [float(elem) for elem in allRelElems]
                allRelElems = numpy.asarray(allRelElems)
                dataVectors[lineNr-1] = allRelElems
            
    # print "variableNames = "
    # print variableNames
    
    # dummy cluster labels
    clusterLabels = numpy.ones(numberOfVariables, dtype = numpy.int)
    return dataVectors, clusterLabels, 1




# TOWN TOWNNO TRACT      LON     LAT MEDV CMEDV     CRIM    ZN INDUS CHAS    NOX    RM   AGE     DIS RAD TAX PTRATIO      B LSTAT
# data from "boston.c" in R library "spData"
def loadBostonHousing(pathprefix):

    filename = pathprefix + "datasets/bostonHousing.csv"
    
    usedVariables = ["CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
    
    # usedVariables = ["CMEDV", "CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
    # usedVariables = ["CMEDV", "CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS", "RAD", "PTRATIO", "B", "LSTAT"]
    # usedVariables = ["CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS", "RAD", "PTRATIO", "B"] #  "LSTAT"]
    
    numberOfSamples = 506
    numberOfVariables = len(usedVariables)
    
    dataVectors = numpy.zeros(shape = (numberOfSamples, numberOfVariables))
    
    nameToCSVIdMapping = {}
    
    with open(filename,'r') as f:
        for lineNr, elemsInLine in enumerate(csv.reader(f)):
            
            if lineNr == 0:
                # get all variable names
                for idInCSV, variableName in enumerate(elemsInLine):
                    if variableName in usedVariables:
                        nameToCSVIdMapping[variableName] = idInCSV
                
                
            else:
                selectedEntries = numpy.zeros(numberOfVariables)
                for i, variableName in enumerate(usedVariables):
                    assert(variableName in nameToCSVIdMapping.keys())
                    selectedEntries[i] = float(elemsInLine[nameToCSVIdMapping[variableName]])
                
                dataVectors[lineNr-1] = selectedEntries
            
    
    # print "dataVectors = "
    # print dataVectors
    
    # dummy cluster labels
    clusterLabels = numpy.ones(numberOfVariables, dtype = numpy.int)
    return dataVectors, clusterLabels, 1


# Gene function regulations data from Kei Hirose used in "Robust Sparse Gaussian Graphical Modeling"
def loadGeneRegulations(pathprefix):

    filename = pathprefix + "datasets/gene_regulations.csv"
    
    numberOfSamples = 445
    numberOfVariables = 11
    
    dataVectors = numpy.zeros(shape = (numberOfSamples, numberOfVariables))
    
    with open(filename,'r') as f:
        for lineNr, elemsInLine in enumerate(csv.reader(f)):
            
            # print len(elemsInLine)
            assert(len(elemsInLine) == numberOfVariables + 1)
            allRelElems = elemsInLine[1:numberOfVariables + 1]
            
            if lineNr == 0:
                # get all variable names
                variableNames = numpy.asarray(allRelElems)
                
            else:
                allRelElems = [float(elem) for elem in allRelElems]
                allRelElems = numpy.asarray(allRelElems)
                dataVectors[lineNr-1] = allRelElems
            
    # print "variableNames = "
    # print variableNames
    # assert(False)
    
    # print "dataVectors = "
    # print dataVectors
    
    # dummy cluster labels
    clusterLabels = numpy.ones(numberOfVariables, dtype = numpy.int)
    return dataVectors, clusterLabels, 1


# BASE_FOLDER = "/Users/danielandrade/workspace/StanTest/"
# dataVectorsAllOriginal, hiddenVarIds, numberOfClusters = loadGeneRegulations(BASE_FOLDER)
# 
# print "dataVectorsAllOriginal.shape = ", dataVectorsAllOriginal.shape
# dataVectorsAllOriginal
# 
# numpy.savetxt(BASE_FOLDER + "datasets/test.csv", dataVectorsAllOriginal, delimiter=",")
# 
# assert(False)


# get node labels for "aviationSuperLargeSmallVar"
def getAviationNodeLabels(pathprefix):
    
    allRelevantKeysDescriptions = numpy.load(pathprefix + "/datasets/relevantDescriptionsAviationData_allOver500.npy")
    selectedVarIdsStr = "54 6 56 61 2 55 18 19 58 8 4 3 1 63 9 64"
    allRelevantKeysDescriptions = allRelevantKeysDescriptions[getIdsFromString(selectedVarIdsStr)]
    assert(allRelevantKeysDescriptions.shape[0] == 16)
    
    return allRelevantKeysDescriptions

def getGeneRegulationsNodeLabels(pathprefix):
    numberOfVariables = 11
    variableNames = None
    with open(pathprefix + "datasets/gene_regulations.csv",'r') as f:
        for lineNr, elemsInLine in enumerate(csv.reader(f)):
            allRelElems = elemsInLine[1:numberOfVariables + 1]
            if lineNr == 0:
                # get all variable names
                variableNames = numpy.asarray(allRelElems)
                break
    return variableNames


def showGeneRegulationsClusteringResult(pathprefix, clusteringResult):
    if clusteringResult is None:
        return
            
    variableNames = getGeneRegulationsNodeNames(pathprefix)           
    print(variableNames)
               
    numberOfClusters = numpy.max(clusteringResult)
    for clusterId in range(1, numberOfClusters+1, 1):
        ids = numpy.where(clusteringResult == clusterId)[0]
        print("Cluster " + str(clusterId) + " = " + ",".join(variableNames[ids]))
        assert(len(ids) >= 1)
        
def colorMFClustering(clusteringResult):
    # \textbf{\color{blue}{2}, \color{red}{2}, \color{brown}{2}, \color{teal}{2}}
    
    USbondFunds = 13
    USstockFunds = 30
    balancedFunds = 7
    internationalStockFunds = 9
    
    coloredClusteringOutput = "\\color{blue}{U.S. bond funds} &  \\textbf{ "
    coloredClusteringOutput += "\\color{blue}{ "
    for varId in range(0, USbondFunds, 1):
        coloredClusteringOutput += str(clusteringResult[varId]) + " "
    
    coloredClusteringOutput += " } } \\\\"
    print(coloredClusteringOutput)
    
    coloredClusteringOutput = "\\color{red}{U.S. stock funds} &  \\textbf{ "
    coloredClusteringOutput += "\\color{red}{ "
    
    for varId in range(USbondFunds, USbondFunds + USstockFunds, 1):
        coloredClusteringOutput += str(clusteringResult[varId]) + " "
    
    coloredClusteringOutput += " } } \\\\"
    print(coloredClusteringOutput)
    
    coloredClusteringOutput = "\\color{brown}{balanced funds} &  \\textbf{ "
    coloredClusteringOutput += "\\color{brown}{ "
    
    for varId in range(USbondFunds + USstockFunds, balancedFunds + USbondFunds + USstockFunds, 1):
        coloredClusteringOutput += str(clusteringResult[varId]) + " "
    
    coloredClusteringOutput += " } } \\\\"
    print(coloredClusteringOutput)
    
    coloredClusteringOutput = "\\color{teal}{international stock funds} &  \\textbf{ "
    coloredClusteringOutput += "\\color{teal}{ "
    
    for varId in range(balancedFunds + USbondFunds + USstockFunds, internationalStockFunds + balancedFunds + USbondFunds + USstockFunds, 1):
        coloredClusteringOutput += str(clusteringResult[varId]) + " "
        
    coloredClusteringOutput += " } } \\\\"
    print(coloredClusteringOutput)
    
    return

# Galactose utilization data from Kei Hirose used in Robust Sparse Gaussian Graphical Modeling"
def loadGalactose(pathprefix):

    filename = pathprefix + "datasets/galactose_utilization.csv"
    
    REMOVE_OUTLIERS = True
    
    if REMOVE_OUTLIERS:
        numberOfSamples = 125
    else:
        numberOfSamples = 136
    
    numberOfVariables = 8
    
    dataVectors = numpy.zeros(shape = (numberOfSamples, numberOfVariables))
    
    sampleCount = 0
    with open(filename,'r') as f:
        for lineNr, elemsInLine in enumerate(csv.reader(f)):
            
            assert(len(elemsInLine) == numberOfVariables)
            
            if lineNr == 0:
                # get all variable names
                variableNames = numpy.asarray(elemsInLine)
                
            else:
                if "NA" not in elemsInLine:
                    allRelElems = [float(elem) for elem in elemsInLine]
                    allRelElems = numpy.asarray(allRelElems)
                    if REMOVE_OUTLIERS and numpy.any(allRelElems < -5.0):
                        print("ignore outlier")
                    else:
                        dataVectors[sampleCount] = allRelElems
                        sampleCount += 1
        
    print("sampleCount = ", sampleCount)
    
    
    # print "variableNames = "
    # print variableNames
    
    # print "dataVectors = "
    # print dataVectors
    # assert(False)
    
    # dummy cluster labels
    clusterLabels = numpy.ones(numberOfVariables, dtype = numpy.int)
    return dataVectors, clusterLabels, 1


# print "dataVectorsAllOriginal = "
# print dataVectorsAllOriginal
# print hiddenVarIds