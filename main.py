#!/usr/bin/env python3

#
# BRAID: A python program to find protein complexes from high-throughput data
# 
# Input file: a CSV file containing a list of bait-preys experiments
# The first protein at the beginning of every line is a bait protein.
#   
# Model parameters:
# 1. False negative rate
# 2. False positive rate
# 3. A maximum possible number of clusters

import argparse
import numpy as np
import matplotlib.pyplot as plt

import spoke_model.countSpokeModel as cpm
import spoke_model.simulateLikelihood as smlt
import spoke_model.simulateLikelihood

class CInputSet:

    def __init__(self, filename):
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

        self.observationG = cpm.CountSpokeModel(nProteins, listBaits, listIndices)

    def writeCluster2File(self, matQ):
        nRows, nCols = matQ.shape
        vecArgMax = np.argmax(matQ,axis=1)
        with open("out.tab", "w") as fh:
            for i in range(nRows):
                fh.write(self.vecProteins[i] + '\t' + str(vecArgMax[i]) + '\n')
            fh.close()

def clustering(inputSet, Nk, psi):
    nProteins = inputSet.observationG.nProteins
    cmfa = smlt.CMeanFieldAnnealing(nProteins, Nk)
    lstExpectedLikelihood = cmfa.Likelihood(inputSet.observationG, nProteins, Nk, psi)
    matQ = cmfa.clusterImage(cmfa.mIndicatorQ)
    print("Number of clusters used: " + str(np.sum(np.sum(matQ, axis=0) > 0)))
    inputSet.writeCluster2File(matQ)
    return lstExpectedLikelihood

def get_args():
    parser = argparse.ArgumentParser(description='MFA')
    parser.add_argument('-f', '--file', metavar='file',
                        default='', help='CSV input file of protein purifications')
    parser.add_argument('-k', '--max', metavar='numclusters',
                        default='100', help='A maximum number of possible clusters')
    parser.add_argument('-psi', '--ratio', metavar='psi',
                        default='3.4', help='A ratio of log(1-fn)/log(1-fp)')
    return parser.parse_args()

def main():
    args = get_args()
    if (args.file == ''):
        print('Input file cannot be empty. Require a CSV file of protein purifications.')
        exit()
    print('Hello, ' + args.file)
    nK = int(args.max)
    psi = float(args.ratio)

    inputSet = CInputSet(args.file)
    nLogLikelihood = clustering(inputSet, nK, psi)
    
    plt.plot(range(len(nLogLikelihood)), nLogLikelihood)
    plt.title('Gavin2002')
    plt.show()

if __name__ == '__main__':
    main()