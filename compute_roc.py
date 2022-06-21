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
from scipy import stats

import spoke_model.countSpokeModel as cpm
import spoke_model.countBioplexSpokeModel as cpmBioplex
import spoke_model.countMatrixModel as cmm
import meanfield.simulateLikelihood as smlt

import inputFile.inputFile as inputFile
import inputFile.inputBioplex as inputBioplex

from time import time as timer

from multiprocessing import Process, Manager

def funcA(inputSet, Nk, psi, dct, l):
    nProteins = inputSet.observationG.nProteins
    cmfa = smlt.CMeanFieldAnnealing(nProteins, Nk)
    cmfa.Likelihood(inputSet.observationG, nProteins, Nk, psi)
    (regr, fn, fp) = cmfa.computeResidues(inputSet.observationG, nProteins, Nk)
    dct[l] = (fn, fp)

def clustering(inputSet, Nk, psi):
    fnrates = []
    fprates = []
    aPsi = np.arange(0.5, 10.0, 1.0)

    with Manager() as manager:
        dct = manager.dict()
        lst = manager.list(range(len(aPsi)))
        for l in lst:
            psi = aPsi[l]
            p = Process(target=funcA, args=(inputSet, Nk, psi, dct, l))
            p.start()
            p.join()

        for d in dct.keys():
            print(dct[d])
            fnrates.append(dct[d][0])
            fprates.append(dct[d][1])
    plt.plot(fnrates, fprates)
    plt.show()

def get_args():
    parser = argparse.ArgumentParser(description='MFA')
    parser.add_argument('-f', '--file', metavar='file',
                        default='', help='CSV input file of protein purifications')
    parser.add_argument('-bp', '--bioplex', action='store_true',
                        default=False, help='Indicate if the input is in Bioplex format')
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

    if args.bioplex:
        inputSet = inputBioplex.CInputBioplex(args.file, cpmBioplex.CountBioplexSpoke)
    else:
        inputSet = inputFile.CInputSet(args.file, cmm.CountMatrixModel)
    clustering(inputSet, nK, psi)

if __name__ == '__main__':
    main()