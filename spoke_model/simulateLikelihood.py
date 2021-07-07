
import numpy as np
from numpy.random import default_rng

import matplotlib.pyplot as plt
import sys

NTrials = 5

# A nice correction suggested by Tomáš Tunys
def Stick_Breaking(num_weights,alpha):
    betas = np.random.beta(1,alpha, size=num_weights)
    betas[1:] *= np.cumprod(1 - betas[:-1])  
    betas /= np.sum(betas)     
    return betas

def Assign_Cluster(rng, betas):
    t = rng.random()
    sorted_arg = np.argsort(betas)
    indicator = np.zeros(len(betas))
    cumulativeSum = 0.0
    for i in range(len(betas)):
        if t >= cumulativeSum and t < (cumulativeSum + betas[sorted_arg[i]]):
            indicator[sorted_arg[i]] = 1
            return indicator
        cumulativeSum += betas[sorted_arg[i]]
    assert(True)
#
# Observation graph
# 
def ObservationGraph(Nproteins, Nk, fn, fp, alpha):
    rng = default_rng()
    #
    # Try stick breaking weights
    # 
    betas = Stick_Breaking(Nk, alpha)
    mIndicatorQ = np.zeros((Nproteins, Nk), dtype=float)
    for i in range(Nproteins):
        mIndicatorQ[i,:] = Assign_Cluster(rng, betas)

    print("Cluster = " + str(mIndicatorQ))
    mObservationSuccess = np.zeros((Nproteins, Nproteins), dtype=int)

    for i in range(Nproteins):
        for j in range(i):
            if (np.argmax(mIndicatorQ[i]) == np.argmax(mIndicatorQ[j])):
                for k in range(NTrials):
                    if( rng.random() > fn ):
                        mObservationSuccess[i][j] += 1
            else:
                for k in range(NTrials):
                    if( rng.random() < fp ):
                        mObservationSuccess[i][j] += 1

    return mObservationSuccess

def Likelihood(mObservationG, Nproteins, Nk, fn, fp):

    rng = default_rng()

    psi = (-np.log(fp) + np.log(1 - fn))/(-np.log(fn) + np.log(1 - fp))
    print('psi = ', psi)

    alpha = 10

    mIndicatorQ = np.zeros((Nproteins, Nk), dtype=float)
    mAlphas = np.ones(Nk, dtype=float)
    mAlphas *= alpha
    for i in range(Nproteins):
        mIndicatorQ[i,:] = np.random.dirichlet(mAlphas)

    gamma = 100
    mLogLikelihood = np.zeros((Nproteins, Nk), dtype=float) # Negative log-likelihood
    
    nLastLogLikelihood = 0.0
    while gamma < 1000:
        for i in range(Nproteins):
            for k in range(Nk):
                for j in mObservationG.lstAdjacency[i]:
                    if (i != j):
                        t = mObservationG.mTrials[i][j]
                        s = mObservationG.mObserved[i][j] 
                        mLogLikelihood[i][k] += (mIndicatorQ[j][k]*(t-s) + (1 - mIndicatorQ[j][k])*s*psi)

        for k in range(Nk):        
            mIndicatorQ[i][k] = np.exp(-gamma*mLogLikelihood[i][k])
        if sum(mIndicatorQ[i,:]) > 0.0:
            mIndicatorQ[i,:] /= sum(mIndicatorQ[i,:])
        gamma *= 2.1
        nEntropy = 0.0
        nLogLikelihood = 0.0
        for i in range(Nproteins):
            for k in range(Nk):
                if (mIndicatorQ[i][k] > 0):
                    nEntropy += mIndicatorQ[i][k]*np.log(mIndicatorQ[i][k])
                    nLogLikelihood += mIndicatorQ[i][k]*mLogLikelihood[i][k]
        if (nLogLikelihood - nLastLogLikelihood < 0.00000001):
            break
        else:
            nLastLogLikelihood = nLogLikelihood

    alpha = -np.log(fn) + np.log(1-fp)
    beta = -np.log(fp) + np.log(1-fn)
    nEntropy = 0.0
    nLogLikelihood = 0.0
    for i in range(Nproteins):
        for k in range(Nk):
            if (mIndicatorQ[i][k] > 0):
                nEntropy += mIndicatorQ[i][k]*np.log(mIndicatorQ[i][k])
                nLogLikelihood += mIndicatorQ[i][k]*mLogLikelihood[i][k]

    print('Entropy = ' + str(nEntropy))
    print('Log-Likelihood = ' + str(nLogLikelihood))
    return nLogLikelihood - nEntropy

if __name__ == '__main__':
    NPROTEINS = 100
    NCLUSTERS = 10
    mGraph = ObservationGraph(NPROTEINS, NCLUSTERS, 0.001, 0.01, 10)

    lstCostFunction = []
    fn = 0.001
    fp = 0.01
    for k in range(2,50):
        minCost = Likelihood(mGraph, NPROTEINS, k, fn, fp)
        lstCostFunction.append(minCost)

    plt.plot(range(2,50), lstCostFunction)
    plt.show()

#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.plot3D(mA[:,0], mA[:,1], mA[:,2])
#plt.show()
