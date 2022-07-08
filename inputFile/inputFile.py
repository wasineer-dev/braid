
import numpy as np

#
# TODO: cpmFunc can be countSpokeModel or countMatrixModel
#
class CInputSet:

    def __init__(self, filename, cpmFunc):
        super().__init__()

        listBaits = list()
        with open(filename) as fh:
            setProteins = set()
            for line in fh:
                lst = line.rstrip().split(',')
                bait = lst[0]
                listBaits.append(bait)
                setProteins = setProteins.union(set(lst))
            print('Number of proteins ' + str(len(setProteins)))
            fh.close()

        self.aSortedProteins = np.sort(np.array(list(setProteins), dtype='U21'))
        bait_inds = np.searchsorted(self.aSortedProteins, np.array(listBaits, dtype='U21'))
        
        print('Number of purifications ' + str(len(bait_inds)))

        nProteins = len(self.aSortedProteins)
        self.incidence = np.zeros(shape=(len(bait_inds), nProteins), dtype=int)
        with open(filename) as fh:
            lineCount = 0
            for line in fh:
                lst = line.rstrip().split(',')
                prey_inds = np.searchsorted(self.aSortedProteins, np.array(lst, dtype='U21'))           
                for id in prey_inds:
                    self.incidence[lineCount][id] = 1
                lineCount += 1
            fh.close()
            
        self.observationG = cpmFunc(nProteins, bait_inds, self.incidence)

    def writeCluster2File(self, matQ, indVec):
        nRows, nCols = matQ.shape
        with open("out.tab", "w") as fh:
            for i in range(nRows):
                ind = indVec[i]
                fh.write(self.aSortedProteins[i] + '\t' + str(indVec[i]) + '\t' + str(max(matQ[ind])) + '\n')
            fh.close()
        with open("out.csv", "w") as fh:
            for k in range(nCols):
                inds = list(i for i in range(nRows) if indVec[i] == k)
                for j in inds:
                    protein = self.aSortedProteins[j].split('__')[0] 
                    fh.write(protein + '\t')
                fh.write('\n')
            fh.close()
    
    def writeLabel2File(self, indVec):
        clusters = {}
        for i,k in enumerate(indVec):
            if k not in clusters.keys():
                clusters[k] = set()
            clusters[k].add(i)

        with open("out.csv", "w") as fh:
            for i, k in enumerate(clusters):
                for v in clusters[k]:
                    protein = self.aSortedProteins[v].split('__')[0] 
                    fh.write(protein + '\t')
                fh.write('\n')
            fh.close()

    def writeCoComplex(self, y_pred):
        Ns, Nk = y_pred.shape
        clusters = {}
        for k in range(Nk):
            clusters[k] = set()
            for i in range(Ns):
                if y_pred[i,k] > 0: clusters[k].add(i)     

        with open("out.csv", "w") as fh:
            for i, k in enumerate(clusters):
                for v in clusters[k]:
                    protein = self.aSortedProteins[v].split('__')[0] 
                    fh.write(protein + '\t')
                fh.write('\n')
            fh.close()
