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
import meanfield.simulateLikelihood as smlt

import inputFile.inputFile as inputFile

def clustering(inputSet, Nk, psi):
    fn = 0.8
    fp = 0.04
    nProteins = inputSet.observationG.nProteins
    cmfa = smlt.CMeanFieldAnnealing(nProteins, Nk)
    lstExpectedLikelihood = cmfa.Likelihood(inputSet.observationG, nProteins, Nk, psi)
    (fn, fp) = cmfa.computeResidues(inputSet.observationG, nProteins, Nk)
    cmfa.computeEntropy(nProteins, Nk)
    matQ = cmfa.clusterImage(cmfa.mIndicatorQ)
    
    print("False negative rate = " + str(fn))
    print("False positive rate = " + str(fp))
    
    inputSet.writeCluster2File(cmfa.mIndicatorQ)
    inputSet.observationG.write2cytoscape(cmfa.mIndicatorQ, inputSet.vecProteins)
    plt.hist(cmfa.mEntropy)
    plt.show()

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

    inputSet = inputFile.CInputSet(args.file, cpm.CountSpokeModel)
    nLogLikelihood = clustering(inputSet, nK, psi)

if __name__ == '__main__':
    main()