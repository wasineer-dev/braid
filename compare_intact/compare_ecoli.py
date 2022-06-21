import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics.cluster import contingency_matrix, pair_confusion_matrix, adjusted_rand_score

setBenchmarkProteins = set()
setObservedProteins = set()

def jaccardIndex(vecA, vecB):
    setA = set()
    for prot in vecA:
        if (prot in setBenchmarkProteins):
            setA.add(prot)
    setB = set()
    for prot in vecB:
        if (prot in setBenchmarkProteins):
            setB.add(prot)
    nIntersect = setA.intersection(setB)
    return float(len(nIntersect))/len(setA.union(setB))

def complexCoverage(vecA, vecB):
    setA = set()
    for prot in vecA:
        if (prot in setBenchmarkProteins):
            setA.add(prot)
    setB = set()
    for prot in vecB:
        if (prot in setBenchmarkProteins):
            setB.add(prot)
    nIntersect = setA.intersection(setB)
    return float(len(nIntersect))/len(setB)

# E.coli, IntAct complex file (tsv format)
# Download on 12.03.2022
def readIntActComplex():
    df = pd.read_table("83333.tsv")
    nRows, nCols = df.shape
    N_COL = 4
    clusters = {}
    for i in range(nRows):
        prots = set()
        for p in df.iloc[i][4].split('|'):
            prots.add(p.split('(')[0])    
        print(df.iloc[i][0], prots)
        clusters[df.iloc[i][0]] = prots
    for k in clusters.keys():
        for prot in clusters[k]:
            setBenchmarkProteins.add(prot)
    return clusters

def mapSymbol2Uniprot():
    df = pd.read_table("uniprot-ecoli.tsv")
    nRows, nCols = df.shape
    uniprots = {}
    for i in range(nRows):
        #gene_symbol = df.iloc[i][1].split("_")[0]
        gene_names = df.iloc[i][2].split(" ")
        for gene in gene_names:
            uniprots[gene] = df.iloc[i][0]
    return uniprots

def readEcoliMFAOutput(fileName):
    uniprots = mapSymbol2Uniprot()
    clusters = {}
    proteins = {}
    with open(fileName) as fh:
        for line in fh:
            lst = line.rstrip().split('\t')
            prot = lst[0]
            #print(prot + '\t' + lst[1])
            if lst[1] not in clusters.keys():
                clusters[lst[1]] = []
            prot = prot.strip()
            if (prot in uniprots.keys()):
                clusters[lst[1]].append(uniprots[prot])
        fh.close()
    print('MRF ' + 'number of complexes = ' + str(len(clusters.keys())))

    predictions = {}
    for k in clusters.keys():
        for prot in clusters[k]:
            setObservedProteins.add(prot)
            predictions[k] = clusters[k]

    with open('symbols_ecoli.txt', 'w') as fh:
        for prot in setObservedProteins:
            p = prot + '_ECOLI'
            fh.write(p + '\n')
        fh.close()
    return predictions

def readEcoliBabu2018():
    uniprots = mapSymbol2Uniprot()
    df = pd.read_table("ecoli_bait_prey_complexes.txt")
    nRows, nCols = df.shape
    clusters = {}
    for i in range(nRows):
        complex = df.iloc[i][0]
        lstProteins = []
        for prot in df.iloc[i][2].strip("\"").split(','):
            if (prot.strip() in uniprots.keys()):
                lstProteins.append(uniprots[prot.strip()])
        # print(complex, '\t', lstProteins)
        if complex not in clusters.keys():
            clusters[complex] = lstProteins

    predictions = {}
    for k in clusters.keys():
        predictions[k] = clusters[k]
    return predictions

def getLabels(prot, clusters):
    for k in clusters.keys():
        if prot in clusters[k]:
            return k
    return None

def listPrediction(setProteins, labels, predicts):
    z_true = list()
    z_pred = list()
    for prot in setProteins:
        true_label = getLabels(prot, labels)
        predict_label = getLabels(prot, predicts)
        if true_label != None and predict_label != None:
            z_true.append(true_label)
            z_pred.append(predict_label)
    return (z_true, z_pred)

def get_args():
    parser = argparse.ArgumentParser(description='MFA')
    parser.add_argument('-f', '--file', metavar='file',
                        default='', help='Output from MFA')
    return parser.parse_args()

def main(args):
    matB = readIntActComplex()
    matA = readEcoliMFAOutput(args.file)
    matC = readEcoliBabu2018()
    
    z_true, z_pred = listPrediction(setObservedProteins, matB, matA)

    m = pair_confusion_matrix(z_true, z_pred)
    print(m)

    print(adjusted_rand_score(z_true, z_pred))

    z_true, z_pred = listPrediction(setObservedProteins, matB, matC)

    m = pair_confusion_matrix(z_true, z_pred)
    print(m)

    print(adjusted_rand_score(z_true, z_pred))

if __name__ == '__main__':
    main(get_args())    

    