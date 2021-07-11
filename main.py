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
# 3. Number of clusters

import argparse
import numpy as np
import matplotlib.pyplot as plt

import spoke_model.countSpokeModel as cpm
import spoke_model.simulateLikelihood as smlt
import spoke_model.simulateLikelihood

def read_input(filename):
    with open(filename) as fh:
        records = dict()
        listInput = []
        for line in fh:
            lst = line.rstrip().split(',')
            listInput.append(lst)
            for protein in lst:
                records[protein] = 0
        states = list(records.keys())
        sorted(states)
        print('Number of proteins ' + str(len(states)))

    nProteins = len(states)
    listBaits = []
    for lst in listInput:
        bait = lst[0]
        listBaits.append(states.index(bait))
    print('Number of purifications ' + str(len(listBaits)))

    listIndices = []
    for lst in listInput:
        indices = []
        for prot in lst:
            if (not states.index(prot) in indices):
                indices.append(states.index(prot))
        listIndices.append(indices)

    observationG = cpm.CountSpokeModel(nProteins, listBaits, listIndices)
    return observationG

def clustering(observationG, Nk, psi):
    nProteins = observationG.nProteins
    cmfa = smlt.CMeanFieldAnnealing(nProteins, Nk)
    lstExpectedLikelihood = cmfa.Likelihood(observationG, nProteins, Nk, psi)
    matQ = cmfa.clusterImage(cmfa.mIndicatorQ)
    print("Number of clusters used: " + str(np.sum(np.sum(matQ, axis=0) > 0)))
    plt.imshow(matQ, interpolation='nearest', aspect='equal')
    plt.show()
    return lstExpectedLikelihood

def get_args():
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('-f', '--file', metavar='file',
                        default='', help='CSV input file of protein purifications')
    parser.add_argument('-k', '--max', metavar='numclusters',
                        default='', help='A maximum number of possible clusters')
    parser.add_argument('-psi', '--ratio', metavar='psi',
                        default='', help='A ratio of log(1-fn)/log(1-fp)')
    return parser.parse_args()

def main():
    args = get_args()
    print('Hello, ' + args.file)
    nK = int(args.max)
    psi = float(args.ratio)

    observationG = read_input(args.file)
    nLogLikelihood = clustering(observationG, nK, psi)
    
    plt.plot(range(len(nLogLikelihood)), nLogLikelihood)
    plt.title('Gavin2002')
    plt.show()

if __name__ == '__main__':
    main()