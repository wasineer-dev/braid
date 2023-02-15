import numpy as np
from time import time as timer

import decimal
from scipy.optimize import minimize, minimize_scalar, approx_fprime
from inputFile import inputFile
from meanfield import simulateLikelihood as smlt

from spoke_model import countMatrixModel as cmm 

inputSet = inputFile.CInputSet("gavin2002.csv", cmm.CountMatrixModel)

Nk = 300
nProteins = inputSet.observationG.nProteins
cmfa = smlt.CMeanFieldAnnealing(nProteins, Nk) # default

decimal.getcontext().prec = 2
decimal.getcontext().rounding = decimal.ROUND_HALF_EVEN
def likelihood_loss(x):
    psi = decimal.Decimal(x)
    mle = cmfa.estimate(inputSet.observationG, nProteins, Nk, psi)
    cmfa.mIndicatorQ = cmfa.tQ.numpy()  
    cmfa.find_argmax()
    (fn, fp, fvalue) = cmfa.computeErrorRate(x, cmfa.indicatorVec, inputSet.observationG, nProteins)
    #inputSet.writeCluster2File("my_dir/out_%.2f.tsv" % x, cmfa.mIndicatorQ, cmfa.indicatorVec)
    #inputSet.observationG.write2cytoscape("my_dir/out_%.2f.sif" % x, cmfa.indicatorVec, cmfa.mIndicatorQ, inputSet.aSortedProteins)    
    regularized_loss = fvalue  # this loss needs to be minimized
    print("psi=%.3f, loss=%.8f" % (psi, regularized_loss))
    return regularized_loss

def gradient(x):
    f1 = likelihood_loss(x)
    f2 = likelihood_loss(x + 0.01)
    g = (decimal.Decimal(f2[0]) - decimal.Decimal(f1[0]))/decimal.Decimal('0.01')
    return (f1[0], float(g))

res = minimize_scalar(likelihood_loss, bounds=(0.5, 8.0), tol=0.1, options={'xatol': 0.1})
print(res.x, res.fun)