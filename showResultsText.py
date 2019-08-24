import numpy
import shared.idcHelper as idcHelper

def getInclusionProbabilities(p, assignmentsWithFrequency):

    varAssignmentCount = numpy.zeros(p)
    
    totalModelCount = 0.0
    for modelId in range(len(assignmentsWithFrequency)):
        modelVars = getNumpyArray(assignmentsWithFrequency[modelId][0])
        modelCount = assignmentsWithFrequency[modelId][1]
        varAssignmentCount[modelVars] += modelCount
        totalModelCount += modelCount
    
    inclusionProbabilities = varAssignmentCount / totalModelCount
    return inclusionProbabilities


def getNumpyArray(assignmentStr):
    if len(assignmentStr) == 0:
        assignmentNumpyArray = numpy.asarray([], dtype = numpy.int)
    else:
        assignmentStrSplit = (assignmentStr).split(" ")
        assignmentNumpyArray = numpy.asarray([int(s) for s in assignmentStrSplit], dtype = numpy.int)
    return assignmentNumpyArray


def showTopN_withModelProbabilities(sortedAssignmentsByFrequency, variableNames, topN):
    totalCount = 0.0
    for elem in sortedAssignmentsByFrequency:
        totalCount += elem[1]
    
    coveredProbability = 0.0
    for i in range(topN):
        elem = sortedAssignmentsByFrequency[i]
        selectedVars = getNumpyArray(elem[0])
        
        probability = elem[1] / totalCount
        coveredProbability += probability
        
        print(idcHelper.getSelectedVariablesNiceStr(variableNames, selectedVars) + " & " + str(round(probability,3)) + " \\\\")
        
    
    print("coveredProbability = " + str(round(coveredProbability,3)))
    return


def showTopN_inclusionProbabilities(p, sortedAssignmentsByFrequency, variableNames, topN):
    inclusionProbabilities = getInclusionProbabilities(p, sortedAssignmentsByFrequency)
    topIds = numpy.argsort(-inclusionProbabilities)[0:topN]
    for varId in topIds:
        print(variableNames[varId] + " & " + str(round(inclusionProbabilities[varId],3)) + " \\\\")

    return

def showTopN_inclusionProbabilities_severalMethods(p, sortedAssignmentsByFrequency_all, variableNames, topN):
    inclusionProbabilities_all = []
    for sortedAssignmentsByFrequency in sortedAssignmentsByFrequency_all:
        inclusionProbabilities_all.append(getInclusionProbabilities(p, sortedAssignmentsByFrequency))
        
    topIds = numpy.argsort(-inclusionProbabilities_all[0])[0:topN]
    for varId in topIds:
        line = variableNames[varId] 
        for inclusionProbabilities in inclusionProbabilities_all:
            line += " & " + str(round(inclusionProbabilities[varId],3))
        print(line + " \\\\")
    return