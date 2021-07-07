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
    

class CountSpokeModel:
    
    def __init__(self, nProteins, listBaits, listIndices):

        self.nProteins = nProteins
        self.mObserved = np.zeros(shape=(nProteins, nProteins), dtype=int)
        for indices in listIndices:
            bait = indices[0]
            for j in indices:
                if bait < j:
                    self.mObserved[j][bait] += 1
                else:
                    self.mObserved[bait][j] += 1

        self.mTrials = np.zeros(shape=(nProteins, nProteins), dtype=int)
        nDim = nProteins*nProteins
        a = np.arange(nDim).reshape(nProteins,nProteins)
        
        for bait in listBaits:
            dim = bait
            self.mTrials[bait, 0:dim:1] += 1
            if dim < nProteins:
                rows = np.arange(nProteins-dim) + dim
                cols = [bait]*len(rows)
                self.mTrials[rows, cols] += 1
        
        self.mPreComputed = interactionProbability(0.3, 0.4, 0.01)
        self.mPosterior = np.zeros(shape=(nProteins, nProteins), dtype=float)

        for i in np.arange(nProteins):
            for j in np.arange(i+1):
                t = self.mTrials[i][j]
                s = self.mObserved[i][j]
                self.mPosterior[i][j] =self.mPreComputed[t][s]

        #
        # Create the adjacency list
        #
        self.lstAdjacency = {}
        for i in np.arange(nProteins):
            self.lstAdjacency[i] = []
            for j in np.arange(i+1):
                t = self.mTrials[i][j]
                s = self.mObserved[i][j] 
                if (t > 0):
                    self.lstAdjacency[i].append(j)
    