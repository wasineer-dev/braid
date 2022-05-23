import numpy as np
import pandas as pd
#
# TODO: cpmFunc can be countSpokeModel or countMatrixModel
#
class CInputBioplex:

    def __init__(self, filename, cpmFunc):
        super().__init__()
        df = pd.read_csv(filename, sep='\t')
        bait_set = set()
        prey_set = set()

        for bait in df['bait_symbol']:
            bait_set.add(bait)

        for prey in df['symbol']:
            prey_set.add(prey)

        proteins_set = prey_set.union(bait_set)
        self.vecProteins = list(proteins_set)
        self.observationG = cpmFunc(filename)

    def writeCluster2File(self, matQ, indVec):
        nRows, nCols = matQ.shape
        with open("out.tab", "w") as fh:
            for i in range(nRows):
                ind = indVec[i]
                fh.write(self.vecProteins[i] + '\t' + str(indVec[i]) + '\t' + str(max(matQ[ind])) + '\n')
            fh.close()