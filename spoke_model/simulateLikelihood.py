
import numpy as np
from numpy.random import default_rng

import matplotlib.pyplot as plt

NTrials = 10

# A nice correction suggested by Tomáš Tunys
def Stick_Breaking(num_weights,alpha):
    betas = np.random.beta(1,alpha, size=num_weights)
    betas[1:] *= np.cumprod(1 - betas[:-1])  
    print("Betas sum = " + str(np.sum(betas)))
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

    betas = Stick_Breaking(Nk, alpha)
    mIndicatorQ = np.zeros((Nproteins, Nk), dtype=float)
    for i in range(Nproteins):
        mIndicatorQ[i,:] = Assign_Cluster(rng, betas)

    gamma = 100
    while gamma < 1000:
        mLogLikelihood = np.zeros((Nproteins, Nk), dtype=float) # Negative log-likelihood
        for i in range(Nproteins):
            for k in range(Nk):
                for j in range(i):
                    if (i != j):
                        t = NTrials
                        s = mObservationG[i][j]
                        mLogLikelihood[i][k] += (mIndicatorQ[j][k]*(t-s) + (1 - mIndicatorQ[j][k])*s*psi)

        for k in range(Nk):        
            mIndicatorQ[i][k] = np.exp(-gamma*mLogLikelihood[i][k])
        if sum(mIndicatorQ[i,:]) > 0.0:
            mIndicatorQ[i,:] /= sum(mIndicatorQ[i,:])
        gamma *= 2.1

    # 
    # TODO: Need to compute the cluster indicator before computing a potential function
    #
    mQ = np.zeros(Nproteins)
    for i in range(Nproteins):
        mQ[i] = np.argmax(mIndicatorQ[i,:])

    alpha = -np.log(fn) + np.log(1-fp)
    beta = -np.log(fp) + np.log(1-fn)
    nLikelihood = 0.0
    for i in range(Nproteins):
        for k in range(Nk):
            for j in range(Nproteins):
                if (i != j):
                    Q = 0.0
                    if (mQ[i] == mQ[j]):
	                    Q = 1.0
                    t = NTrials
                    s = mObservationG[i][j]
                    A = s*(Q*(-np.log(1-fn)) + (1.0-Q)*(-np.log(fp)))
                    B = (t-s)*(Q*(-np.log(fn)) + (1.0-Q)*(-np.log(1-fp)))
                    nLikelihood += (A + B)

    print('Likelihood = ' + str(nLikelihood))

    return nLikelihood

NPROTEINS = 1000
NCLUSTERS = 10
mGraph = ObservationGraph(NPROTEINS, NCLUSTERS, 0.001, 0.01, 100)

lstCostFunction = []
for k in range(2,20):
    lstCostFunction.append(Likelihood(mGraph, NPROTEINS, k, 0.001, 0.01))

plt.plot(range(2,20), lstCostFunction)
plt.show()

#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.plot3D(mA[:,0], mA[:,1], mA[:,2])
#plt.show()
