import numpy
import rpy2.robjects as ro
import scipy.special

def getRidgeEstimate(y, X):
    p = X.shape[1]
    invEst = numpy.linalg.inv(X.transpose() @ X + 0.0001 * numpy.eye(p))
    ridgeBetaEst = (invEst @ X.transpose()) @ y
    return ridgeBetaEst

def thresholdBaseline(y, X, delta):
    
    ridgeBetaEst = getRidgeEstimate(y, X)
    # print("ridgeBetaEst = ", ridgeBetaEst)
    # threshold = numpy.max(ridgeBetaEst) * 0.15
    selectedVars = numpy.where(numpy.abs(ridgeBetaEst) > delta)[0]
    # print("selectedVars = ", selectedVars)
    return selectedVars



def runLars(X, y):
    ro.globalenv['X'] = ro.r.matrix(X, nrow = X.shape[0], ncol = X.shape[1])
    ro.globalenv['y'] = ro.r.matrix(y, nrow = y.shape[0], ncol = 1)
    ro.r('larsFit = lars(X, y, intercept = FALSE, normalize = FALSE, use.Gram=TRUE)')
    
    # WARNING: the result for lars is the transpose of the result for glmnet !!!
    allBetaEstimates_LARS = ro.r('as.matrix(coef.lars(larsFit))')
    
    allNonZeroPositions_foundByLARS = []
    for larsEstId in range(allBetaEstimates_LARS.shape[0]):
        betaEstimate = allBetaEstimates_LARS[larsEstId]
        nonZeroPositions = numpy.where(betaEstimate != 0)[0]
        allNonZeroPositions_foundByLARS.append(nonZeroPositions)
    
    return allNonZeroPositions_foundByLARS
    
    
def runEMVS(X, y):
    ro.globalenv['X'] = ro.r.matrix(X, nrow = X.shape[0], ncol = X.shape[1])
    ro.globalenv['y'] = ro.r.matrix(y, nrow = y.shape[0], ncol = 1)
    
    # same as in example setting of EMVS
    ro.r('v0 = seq(0.1, 2, length.out = 20)')
    ro.r('v1 = 1000')
    
    # ro.r('v0 = seq(0.1, 2, length.out = 20)')
    # ro.r('v0 = v0 / 1000')
    # ro.r('v1 = 100')
    
    ro.r('beta_init = rep(1, ' + str(X.shape[1]) + ')')
          
    ro.r('sigma_init = 1')
    ro.r('a = b = 1')
    ro.r('epsilon = 10^{-5}')
    ro.r('result = EMVS(y, X, v0 = v0, v1 = v1, type = "betabinomial",  beta_init = beta_init, independent = FALSE, sigma_init = sigma_init, epsilon = epsilon, a = a, b = b, standardize = TRUE)')
    selectedVars = ro.r('as.matrix(EMVSbest(result)$indices)')
    
    
    selectedVars = selectedVars - 1   # since R starts incidices counting from 1
    selectedVars = selectedVars[:,0]
    
    # print("selectedVars = ")
    # print(selectedVars)
    return selectedVars

def runSSLASSO(X, y):
    ro.globalenv['X'] = ro.r.matrix(X, nrow = X.shape[0], ncol = X.shape[1])
    ro.globalenv['y'] = ro.r.matrix(y, nrow = y.shape[0], ncol = 1)
    
    ro.r('result = SSLASSO(X, y, penalty = "adaptive", variance = "unknown")')
    selectedVars = ro.r('as.matrix(result$model)')
    
    selectedVars = selectedVars - 1   # since R starts incidices counting from 1
    selectedVars = selectedVars[:,0]
    
    # print("selectedVars = ")
    # print(selectedVars)
    return selectedVars


def runHorseshoe(X, y, MCMC_SAMPLES):
    ro.globalenv['BURN_IN_SAMPLES'] = int(MCMC_SAMPLES * 0.1)
    ro.globalenv['MCMC_SAMPLES'] = MCMC_SAMPLES - int(MCMC_SAMPLES * 0.1)
    
    ro.globalenv['X'] = ro.r.matrix(X, nrow = X.shape[0], ncol = X.shape[1])
    ro.globalenv['y'] = ro.r.matrix(y, nrow = y.shape[0], ncol = 1)
    ro.r('horseshoeResults <- horseshoe(y, X, method.tau = "truncatedCauchy", method.sigma = "Jeffreys", burn = BURN_IN_SAMPLES, nmc = MCMC_SAMPLES)')
    Sigma2Hat = ro.r('horseshoeResults$Sigma2Hat')[0]
    ro.r('selectedVars <- HS.var.select(horseshoeResults, y, method = "intervals")')
    selectedVarsIndicators = ro.r('as.matrix(selectedVars)')
    meanBeta = ro.r('as.matrix(horseshoeResults$BetaHat)')
    selectedVars = numpy.where(selectedVarsIndicators == 1.0)[0]
    return selectedVars, meanBeta.transpose()[0], Sigma2Hat

# uses the runSpikeSlabGAM package to run the NMIG model
# Ishwaran and Rao (2005): The basic prior structure, which we call a Normal - mixture of inverse Gammas (NMIG) prior
def runSpikeSlabGAM(X, y):
    p = X.shape[1]
    ro.globalenv['X'] = ro.r.matrix(X, nrow = X.shape[0], ncol = X.shape[1])
    ro.globalenv['y'] = ro.r.matrix(y, nrow = y.shape[0], ncol = 1)
    
    ro.globalenv['p'] = p
    ro.r('colnames(X) <- paste("x", 1:p, sep = "")')
    ro.r('myDataFrame <- data.frame(y, X)')
    
    # create formula string like y ~ lin(x1) + lin(x2) + lin(x3)
    formulaStr = "y ~ "
    formulaStr += " + ".join(["lin(x" + str(i) +")" for i in range(1,X.shape[1] + 1)])
    
    ro.r('options(mc.cores = 5)')
    
    ro.r('mcmc <- list(nChains = 8, chainLength = 1000, burnin = 500, thin = 5)')
    ro.r('m0 <- spikeSlabGAM(' + formulaStr + ', family = "gaussian", data = myDataFrame, mcmc = mcmc)')
    
    ro.r('summaryStatistics <- summary(m0)')
    inclusionProbabilities = ro.r('summaryStatistics[13]$postMeans$pV1')
    # print(inclusionProbabilities)
    
    selectedVars = numpy.where(inclusionProbabilities > 0.5)[0]
    
    # print(ro.r('summaryStatistics'))
    # print(selectedVars)
    return selectedVars


def stabilitySelection(X, y, qValue):
    n = X.shape[0]
    p = X.shape[1]
    ro.globalenv['X'] = ro.r.matrix(X, nrow = X.shape[0], ncol = X.shape[1])
    ro.globalenv['y'] = ro.r.matrix(y, nrow = y.shape[0], ncol = 1)
    
    # qValue = numpy.min([p / 2.0, 50])
    # qValue = numpy.min([6, 100])
    # qValue = 6
    
    ro.globalenv['q'] = qValue
    print("q is set to " + str(ro.globalenv['q']))
    print("START stability selection with glmnet.lasso")
    
    # from documentation in "stabs" package
    # q = number of (unique) selected variables (or groups of variables depending on the model) that are selected on each subsample.
    
    ro.r('stabLasso <- stabsel(X, y, fitfun = glmnet.lasso, q = q, PFER = 1)')
    
    print("FINISHED stability selection with glmnet.lasso")
    
    selectedVars = ro.r('stabLasso$selected')
    selectedVars = selectedVars - 1  # since R starts incidices counting from 1
    return selectedVars

    
def runGibbsBvs(X, y, MCMC_SAMPLES):
    p = X.shape[1]
    ro.globalenv['X'] = ro.r.matrix(X, nrow = X.shape[0], ncol = X.shape[1])
    ro.globalenv['y'] = ro.r.matrix(y, nrow = y.shape[0], ncol = 1)
    
    ro.globalenv['p'] = p
    ro.r('colnames(X) <- paste("x", 0:(p-1), sep = "")')
    ro.r('myDataFrame <- data.frame(y, X)')
    
    ro.globalenv['BURN_IN_SAMPLES'] = int(MCMC_SAMPLES * 0.1)
    ro.globalenv['MCMC_SAMPLES'] = MCMC_SAMPLES - int(MCMC_SAMPLES * 0.1)
    
    
    from rpy2.robjects.packages import importr
    importr('BayesVarSel')
    
    
    if X.shape[0] >= 1000:
        modelName = "\"gZellner\"" 
    else:
        modelName = "\"Robust\"" # prior proposed in "Criteria for Bayesian Model choice with Application to Variable Selection.", The Annals of Statistics 
    
    
    # modelName = "\"Liangetal\""  # prior proposed in "Mixtures of g-priors for Bayesian Variable Selection. "
    
    ro.r('''result <- GibbsBvs(formula= "y ~ .", data=myDataFrame, prior.betas=''' + modelName + ''', 
            n.iter=MCMC_SAMPLES, init.model="Full", n.burnin=BURN_IN_SAMPLES,
            time.test = FALSE)''')
    
#     ro.r('''result <- GibbsBvs(formula= "y ~ .", data=myDataFrame, prior.betas=''' + modelName + ''', 
#             n.iter=MCMC_SAMPLES, init.model="Full", n.burnin=BURN_IN_SAMPLES,
#             time.test = FALSE)''')
    
    selectedVars = numpy.where(ro.r('result$HPMbin') == 1)[0]
    # print("selectedVars = ")
    # print(selectedVars)
    # print("MCMC_SAMPLES = ", ro.globalenv['MCMC_SAMPLES'])
    # print("BURN_IN_SAMPLES = ", ro.globalenv['BURN_IN_SAMPLES'])
    # assert(False)
    return selectedVars


def runZellnerPriorVariableSelection(X, y, allNonZeroPositions_foundByLARS):
    p = X.shape[1]
    ro.globalenv['X'] = ro.r.matrix(X, nrow = X.shape[0], ncol = X.shape[1])
    ro.globalenv['y'] = ro.r.matrix(y, nrow = y.shape[0], ncol = 1)
    
    ro.globalenv['p'] = p
    ro.r('colnames(X) <- paste("x", 0:(p-1), sep = "")')
    ro.r('myDataFrame <- data.frame(y, X)')
    
    print("allNonZeroPositions_foundByLARS = ")
    allModelListStr = []
    allPrioProbs = []
    for i, non_zero_pos in enumerate(allNonZeroPositions_foundByLARS):
        formulaStr = "model" + str(i) + " <- y ~ -1 " 
        if len(non_zero_pos) > 0:
            formulaStr += " + "
        formulaStr += " + ".join(["x" + str(pos) for pos in non_zero_pos])
        ro.r(formulaStr)
        allModelListStr.append("model" + str(i))
        
        # add prior 
        a = 1.0
        b = 1.0
        s = non_zero_pos.shape[0]
        priorLogProb = scipy.special.betaln(a + s, b + p - s) - scipy.special.betaln(a, b)
        priorProb = numpy.exp(priorLogProb)
        allPrioProbs.append(str(priorProb))


    defineAllPriorProbsStr = 'allPriorProbs <- list(' + ",".join(allPrioProbs)  + ')'
    defineAllModelsStr = 'allModels <- list(' + ",".join(allModelListStr) + ')'
    ro.r(defineAllPriorProbsStr)
    ro.r(defineAllModelsStr)
    
    # print(defineAllPriorProbsStr)
    # assert(False)
    # ro.r('resultZellner <- Btest(models=allModels, data=myDataFrame)')
    
    if p < 100:
        ro.r('resultZellner <- Btest(models=allModels, data=myDataFrame, prior.models = \"User\", priorprobs=allPriorProbs)')
    else:
        ro.r('resultZellner <- Btest(models=allModels, data=myDataFrame,  prior.models = \"User\", priorprobs=allPriorProbs, prior.betas = \"Liangetal\")')
    
#     if y.shape[0] <= 10000:
#     else:
#         # use normal g-Zellner prior because the default has infinite bayes factor
#         ro.r('resultZellner <- Btest(models=allModels, data=myDataFrame, prior.betas = \"gZellner\")')
        
    posteriorProbs = ro.r('resultZellner$PostProbi')
    assert(posteriorProbs.shape[0] == len(allNonZeroPositions_foundByLARS))
    return numpy.log(posteriorProbs)

    # print(posteriorProbs)
    # bestModelId = numpy.argmax(posteriorProbs)
    # print("bestModelId = ", bestModelId)
    # assert(False)
    

