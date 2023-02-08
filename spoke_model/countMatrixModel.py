#
# Generate the observation count using the Spoke model
#

import numpy as np
import scipy.special

def trueInteraction(t, s, fnRate):
    return scipy.special.binom(t,s)*np.power(fnRate,t-s)*np.power(1.0 - fnRate,s)

def falseInteraction(t, s, fpRate):
    return scipy.special.binom(t,s)*np.power(1.0 - fpRate,t-s)*np.power(fpRate,s)

def interactionProbability(rho, fnRate, fpRate):
    MAX_TRIALS = 100

    mPreComputed = np.zeros(shape=(MAX_TRIALS,MAX_TRIALS), dtype=float)
    for t in np.arange(MAX_TRIALS):
        for s in np.arange(t+1):
            mPreComputed[t][s] = np.log(trueInteraction(t,s,fnRate))
            mPreComputed[t][s] -= np.log(trueInteraction(t,s,fnRate)*rho + falseInteraction(t,s,fpRate)*(1.0 - rho))

    return mPreComputed
    

class CountMatrixModel:
    
    def __init__(self, nProteins, bait_inds, incidence):

        self.nProteins = nProteins
        self.mObserved = np.zeros(shape=(nProteins, nProteins), dtype=int)
        for i, bait in zip(range(len(bait_inds)), bait_inds):
            for j in range(nProteins):
                if incidence[i,j]:
                    self.mObserved[j,:] += incidence[i,:] 
                    self.mObserved[:,j] += incidence[i,:]
    
        self.mTrials = np.zeros(shape=(nProteins, nProteins), dtype=int)
        for i, bait in zip(range(len(bait_inds)), bait_inds):
            for j in range(nProteins):
                if incidence[i,j]:
                    self.mTrials[j,:] += np.ones(nProteins, dtype=int) 
                    self.mTrials[:,j] += np.ones(nProteins, dtype=int)

        for i in range(nProteins):
            assert(np.sum(self.mTrials[i,:]) == np.sum(self.mTrials[:,i]))

        #
        # Create the adjacency list
        #
        self.lstAdjacency = {}
        for i in np.arange(nProteins):
            self.lstAdjacency[i] = set()
            for j in np.arange(nProteins):
                t = self.mTrials[i][j]
                if (i < j):
                    s = self.mObserved[i][j] 
                else:
                    s = self.mObserved[j][i] 
                assert(s <= t)
                if (i != j and t > 0):
                    self.lstAdjacency[i].add(j)
                    
    def write2cytoscape(self, fileName, indicators, matQ, vecProteins):
        nRows, nCols = matQ.shape
        with open(fileName, "w") as fh:
            for k in range(nCols):
                inds = list(i for i in range(nRows) if indicators[i] == k)
                for i in inds:
                    for j in inds:
                        t = self.mTrials[i][j]
                        if (i >= j):
                            continue
                        if (t > 0):
                            fh.write(vecProteins[i] + '\t' + str(indicators[i]) + '\t' + str(vecProteins[j]) + '\n')
            fh.close()