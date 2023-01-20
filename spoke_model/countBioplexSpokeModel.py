import pandas as pd
import numpy as np

class CountBioplexSpoke:

    def __init__(self, filePath, bait_inds, incidence):

        Nd, Np = incidence.shape
        nProteins = Np
        self.nProteins = Np
        self.mObserved = np.zeros(shape=(nProteins, nProteins), dtype=int)
        for i, bait in zip(range(len(bait_inds)), bait_inds):
            self.mObserved[i,:] += incidence[i,:] 
            self.mObserved[:,i] += incidence[i,:]
        
        self.mTrials = np.zeros(shape=(nProteins, nProteins), dtype=int)
        for i, bait in zip(range(len(bait_inds)), bait_inds):
            self.mTrials[i,:] += np.ones(nProteins, dtype=int) 
            self.mTrials[:,i] += np.ones(nProteins, dtype=int)
            
        for i in range(nProteins):
            assert(np.sum(self.mTrials[i,:]) == np.sum(self.mTrials[:,i]))
        
        #
        # Create the adjacency list
        #
        self.lstAdjacency = {}
        for i in np.arange(nProteins):
            self.lstAdjacency[i] = []
            for j in np.arange(nProteins):
                t = self.mTrials[i][j]
                if (i < j):
                    s = self.mObserved[i][j] 
                else:
                    s = self.mObserved[j][i] 
                if (i != j and t > 0):
                    assert(s <= t)
                    self.lstAdjacency[i].append(j)

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