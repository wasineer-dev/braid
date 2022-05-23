import pandas as pd
import numpy as np

class CountBioplexSpoke:

    def __init__(self, filePath):

        df = pd.read_csv(filePath, sep='\t')

        bait_set = set()
        prey_set = set()

        for bait in df['bait_symbol']:
            bait_set.add(bait)

        for prey in df['symbol']:
            prey_set.add(prey)

        proteins_set = prey_set.union(bait_set)

        print('Number of baits = ', len(bait_set))
        print('Number of preys = ', len(prey_set))
        print('Number of proteins = ', len(proteins_set))
            
        self.nProteins = len(proteins_set)
        nProteins = len(proteins_set)
        self.mObserved = np.zeros(shape=(nProteins, nProteins), dtype=int)
        
        proteins_list = list(proteins_set)
        nrows, ncols = df.shape
        for row in np.arange(nrows):
            bait = df.loc[row,'bait_symbol']
            prey = df.loc[row,'symbol']
            i = proteins_list.index(bait)
            j = proteins_list.index(prey)
            assert(i >= 0 and j >= 0)
            self.mObserved[i][j] += 1
            self.mObserved[j][i] += 1

        self.mTrials = np.zeros(shape=(nProteins, nProteins), dtype=int)
        for row in np.arange(nrows):
            bait = df.loc[row,'bait_symbol']
            i = proteins_list.index(bait)
            for j in range(nProteins):
                self.mTrials[i][j] += 1
                self.mTrials[j][i] += 1
        
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