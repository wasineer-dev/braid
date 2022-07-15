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
from sklearn import mixture

import spoke_model.countSpokeModel as cpm
import spoke_model.countBioplexSpokeModel as cpmBioplex
import spoke_model.countBioplexMatrixModel as cmmBioplex
import spoke_model.countMatrixModel as cmm
import meanfield.simulateLikelihood as smlt
import mixmodel.mixtureBernoulli as mmb
import mixmodel.betaProcess as mbp

import inputFile.inputFile as inputFile
import inputFile.inputBioplex as inputBioplex

from time import time as timer

def clustering(inputSet, Nk, psi):
    fn = 0.8
    fp = 0.04
    nProteins = inputSet.observationG.nProteins
    cmfa = smlt.CMeanFieldAnnealing(nProteins, Nk) # default

    funcInfer = cmfa

    ts = timer()
    # alpha = 1e-2
    funcInfer.estimate(inputSet.observationG, nProteins, Nk, psi) 
    te = timer()
    print("Time running MFA: ", te-ts)
    (fn, fp) = funcInfer.computeResidues(inputSet.observationG, nProteins, Nk)
    
    print("False negative rate = " + str(fn))
    print("False positive rate = " + str(fp))
    inputSet.writeCluster2File(funcInfer.mIndicatorQ, funcInfer.indicatorVec)
    inputSet.observationG.write2cytoscape(funcInfer.indicatorVec, funcInfer.mIndicatorQ, inputSet.aSortedProteins)

    """
    X = funcInfer.expectedErrors
    y = funcInfer.mResidues
    pred_ols = regr.get_prediction()
    iv_l = pred_ols.summary_frame()["obs_ci_lower"]
    iv_u = pred_ols.summary_frame()["obs_ci_upper"]

    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax1.plot(X, y, "o", label="data")
    ax1.plot(X, regr.fittedvalues, "r--.", label="OLS")
    ax1.plot(X, iv_u, "r--")
    ax1.plot(X, iv_l, "r--")
    ax1.legend(loc="best")
    plt.show()
    
    fig, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(regr.fittedvalues, regr.resid_pearson)
    ax2.hlines(0, 0, np.max(regr.fittedvalues))
    #ax2.set_xlim(0, 1)
    ax2.set_title('Residual Dependence Plot')
    ax2.set_ylabel('Pearson Residuals')
    ax2.set_xlabel('Fitted values')
    plt.show()

    fig, ax3 = plt.subplots(figsize=(8,6))
    ax3.plot(range(Nk), funcInfer.mPosteriorWeights)
    ax3.set_title('Prior weights')
    plt.show()
    """
    return 0

def mixture_bernoulli(inputSet, Nk, psi):
    Xs = np.transpose(inputSet.incidence)
    mb = mmb.MixtureBernoulli(inputSet.observationG, psi)
    p, mix_p = mb.estimate(Xs, Nk, 1e-8, 1e-8)
    y_pred = mb.predict(Xs, p, mix_p)
    inputSet.writeLabel2File(y_pred)

def beta_process(inputSet, Nk, psi):
    Xs = np.array(inputSet.incidence, dtype=float)
    Nd, Ns = Xs.shape
    Xs += np.random.normal(size=(Ns*Nd)).reshape(Nd,Ns)       
    mb = mbp.BetaProcess(inputSet.observationG, Xs, Nk)
    mb.estimate(Xs, Nk)
    y_pred = mb.predict(Xs)
    inputSet.writeCoComplex(y_pred)

def get_args():
    parser = argparse.ArgumentParser(description='MFA')
    parser.add_argument('-f', '--file', metavar='file',
                        default='', help='CSV input file of protein purifications')
    parser.add_argument('-bp', '--bioplex', metavar='bioplex',
                        default='', help='Indicate if the input is in Bioplex format')
    parser.add_argument('-mm', '--mixmodel', metavar="mixmodel",
                        default='none', help='switch to mixture model')
    parser.add_argument('-k', '--max', metavar='numclusters',
                        default='100', help='A maximum number of possible clusters')
    parser.add_argument('-psi', '--ratio', metavar='psi',
                        default='3.4', help='A ratio of log(1-fn)/log(1-fp)')
    return parser.parse_args()

def main():
    args = get_args()
    if (args.file == '' and args.bioplex == ''):
        print('Input file cannot be empty. Require a CSV file of protein purifications, or BioPlex 2.0/3.0 input file.')
        exit()
    
    nK = int(args.max)
    psi = float(args.ratio)

    if args.bioplex != '':
        print('Hello, ' + args.bioplex)
        inputSet = inputBioplex.CInputBioplex(args.bioplex, cmmBioplex.CountBioplexMatrix)
    else:
        print('Hello, ' + args.file)
        inputSet = inputFile.CInputSet(args.file, cmm.CountMatrixModel)
    
    if args.mixmodel == "none":
        return clustering(inputSet, nK, psi)

    if args.mixmodel == "bernoulli":
        return mixture_bernoulli(inputSet, nK, psi)

    if args.mixmodel == "beta":
        return beta_process(inputSet, nK, psi)

if __name__ == '__main__':
    main()