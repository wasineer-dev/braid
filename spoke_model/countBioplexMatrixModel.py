import pandas as pd
import numpy as np

class CountBioplexMatrix:

    def __init__(self, filePath, bait_inds, incidence):
        Nd, Np = incidence.shape
        nProteins = Np
        self.nProteins = Np
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

    def write2cytoscape(self, indicators, matQ, vecProteins):
        nRows, nCols = matQ.shape
        with open("out.sif", "w") as fh:
            for i in np.arange(nRows):
                for j in self.lstAdjacency[i]:
                    t = self.mTrials[i][j]
                    if (i >= j):
                        continue
                    if (t > 0 and indicators[i] == indicators[j]):
                        fh.write(str(vecProteins[i]) + '\t' + str(indicators[i]) + '\t' + str(vecProteins[j]) + '\n')
            fh.close()