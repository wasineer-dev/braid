
import numpy as np
from numpy.random import default_rng

import matplotlib.pyplot as plt
import sys
import scipy.special

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
    NTrials = 5
    #
    # Try stick breaking weights
    # 
    betas = Stick_Breaking(Nk, alpha) # Sample from the Dirichlet distribution
    sizeDistribution = np.random.multinomial(Nproteins, betas, 1)
    lstDistribution = []
    for p in sizeDistribution[0]:
        lstDistribution.append(p/Nproteins)
    mIndicatorQ = np.zeros((Nproteins, Nk), dtype=float)
    for i in range(Nproteins):
        mIndicatorQ[i,:] = Assign_Cluster(rng, lstDistribution)

    print("Cluster = " + str(mIndicatorQ))
    mObservationSuccess = np.zeros((Nproteins, Nproteins), dtype=int)

    for i in range(Nproteins):
        for j in range(i):
            if (np.argmax(mIndicatorQ[i]) == np.argmax(mIndicatorQ[j])):
                mObservationSuccess[i][j] += np.random.binomial(NTrials, 1-fn, 1)
            else:
                mObservationSuccess[i][j] += np.random.binomial(NTrials, fp, 1)

    return mObservationSuccess

class CMeanFieldAnnealing:

    def __init__(self, Nproteins, Nk):
        self.lstExpectedLikelihood = []
        self.mIndicatorQ = np.zeros((Nproteins, Nk), dtype=float)

    def Likelihood(self, mObservationG, Nproteins, Nk, fn, fp):

        rng = default_rng()

        psi = (-np.log(fp) + np.log(1 - fn))/(-np.log(fn) + np.log(1 - fp))
        print('psi = ', psi)

        alpha = 10

        
        mAlphas = np.ones(Nk, dtype=float)
        mAlphas *= alpha
        mComplexDistribution = np.random.dirichlet(mAlphas)
        for i in range(Nproteins):
            self.mIndicatorQ[i,:] = 1. + rng.random(Nk)
            self.mIndicatorQ[i,:] /= sum(self.mIndicatorQ[i,:])

        gamma = 1000.0
        mLogLikelihood = np.zeros((Nproteins, Nk), dtype=float) # Negative log-likelihood

        # TODO: refactor
        # Initialize mLogLikelihood
        for i in range(Nproteins):
            mLogLikelihood[i,:] = 0.0
            for k in range(Nk):
                for j in mObservationG.lstAdjacency[i]:
                    t = mObservationG.mTrials[i][j]
                    s = mObservationG.mObserved[i][j]
                    assert(s <= t)
                    mLogLikelihood[i][k] += (self.mIndicatorQ[j][k]*(t-s) + (1.0 - self.mIndicatorQ[j][k])*s*psi)
            # Overflow problem. Need to compute with softmax
            self.mIndicatorQ[i,:] = scipy.special.softmax(-gamma*mLogLikelihood[i,:])
            self.mIndicatorQ[i,:] /= sum(self.mIndicatorQ[i,:])

        nLastLogLikelihood = 0.0
        nIteration = 0
        while nIteration < 400:
            # TODO: implement multiprocessing in python
            i = np.random.randint(0, Nproteins) # Choose a node at random
            mLogLikelihood[i,:] = 0.0
            for k in range(Nk):
                for j in mObservationG.lstAdjacency[i]:
                    t = mObservationG.mTrials[i][j]
                    s = mObservationG.mObserved[i][j]
                    assert(s <= t)
                    mLogLikelihood[i][k] += (self.mIndicatorQ[j][k]*(t-s) + (1.0 - self.mIndicatorQ[j][k])*s*psi)

            # Overflow problem. Need to compute with softmax
            self.mIndicatorQ[i,:] = scipy.special.softmax(-gamma*mLogLikelihood[i,:])
            self.mIndicatorQ[i,:] /= sum(self.mIndicatorQ[i,:])

            gamma *= 0.1 # decreasing gamma
            nEntropy = 0.0
            nLogLikelihood = 0.0
            for i in range(Nproteins):
                for k in range(Nk):
                    if (self.mIndicatorQ[i][k] > 0):
                        nEntropy += self.mIndicatorQ[i][k]*np.log(self.mIndicatorQ[i][k])
                        nLogLikelihood += self.mIndicatorQ[i][k]*mLogLikelihood[i][k]
            print('Expected log-likelihood = ' + str(nLogLikelihood))
            print('Entropy = ' + str(nEntropy))
            if (np.abs(np.round(nLogLikelihood, decimals=3) - np.round(nLastLogLikelihood, decimals=3)) < 0.001):
                continue
            else:
                nLastLogLikelihood = nLogLikelihood
                self.lstExpectedLikelihood.append(nLogLikelihood)
            nIteration += 1

        alpha = -np.log(fn) + np.log(1-fp)
        beta = -np.log(fp) + np.log(1-fn)
        nEntropy = 0.0
        nLogLikelihood = 0.0
        for i in range(Nproteins):
            for k in range(Nk):
                if (self.mIndicatorQ[i][k] > 0):
                    nEntropy += self.mIndicatorQ[i][k]*np.log(self.mIndicatorQ[i][k])
                    nLogLikelihood += self.mIndicatorQ[i][k]*mLogLikelihood[i][k]

        return self.lstExpectedLikelihood

if __name__ == '__main__':
    NPROTEINS = 100
    NCLUSTERS = 10
    mGraph = ObservationGraph(NPROTEINS, NCLUSTERS, 0.001, 0.01, 10)

    lstCostFunction = []
    fn = 0.001
    fp = 0.01
    for k in range(2,50):
        cmfa = CMeanFieldAnnealing(NPROTEINS, NCLUSTERS)
        minCost = cmfa.Likelihood(mGraph, NPROTEINS, k, fn, fp)
        lstCostFunction.append(minCost)

    plt.plot(range(2,50), lstCostFunction)
    plt.show()

#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.plot3D(mA[:,0], mA[:,1], mA[:,2])
#plt.show()
