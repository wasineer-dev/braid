
import numpy as np

#
# TODO: cpmFunc can be countSpokeModel or countMatrixModel
#
class CInputSet:

    def __init__(self, filename, cpmFunc):
        super().__init__()
        with open(filename) as fh:
            records = dict()
            listInput = []
            for line in fh:
                lst = line.rstrip().split(',')
                listInput.append(lst)
                for protein in lst:
                    records[protein] = 0
            self.vecProteins = list(records.keys())
            sorted(self.vecProteins)
            print('Number of proteins ' + str(len(self.vecProteins)))
            fh.close()

        nProteins = len(self.vecProteins)
        listBaits = []
        for lst in listInput:
            bait = lst[0]
            listBaits.append(self.vecProteins.index(bait))
        print('Number of purifications ' + str(len(listBaits)))

        listIndices = []
        for lst in listInput:
            indices = []
            for prot in lst:
                if (not self.vecProteins.index(prot) in indices):
                    indices.append(self.vecProteins.index(prot))
            listIndices.append(indices)

        self.observationG = cpmFunc(nProteins, listBaits, listIndices)

    def writeCluster2File(self, matQ):
        nRows, nCols = matQ.shape
        vecArgMax = np.argmax(matQ,axis=1)
        with open("out.tab", "w") as fh:
            for i in range(nRows):
                fh.write(self.vecProteins[i] + '\t' + str(vecArgMax[i]) + '\t' + str(matQ[i][vecArgMax[i]]) + '\n')
            fh.close()