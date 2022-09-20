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

def mapSymbol2Uniprot():
    df = pd.read_table("uniprot-bioplex20.tab")
    nRows, nCols = df.shape
    uniprots = {}
    for i in range(nRows):
        gene_symbol = df.iloc[i][0].split("_")[0]
        prot = df.iloc[i][1]
        gene = df.iloc[i][0]
        uniprots[gene_symbol] = prot
    return uniprots

def readClusterOne(fileName):
    uniprots = mapSymbol2Uniprot()
    with open(fileName) as fh:
        fh2 = open("cl1_bioplex.txt", "w")
        for line in fh:
            lst = line.rstrip().split('\t')
            cluster = []
            for prot in lst:
                prot = prot.strip()
                if (prot in uniprots.keys()):
                    cluster.append(uniprots[prot])
                    fh2.write(uniprots[prot] + '\t')
            fh2.write("\n")        
        fh2.close()
        fh.close()

def readBioPlexMFAOutput(fileName):
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

    with open("bioplex20_mrf_complexes.txt", "w") as fh:
        for k, cl in enumerate(clusters):
            for prot in clusters[cl]:
                fh.write(prot + '\t')
            fh.write('\n')
        fh.close()
    return predictions

def get_args():
    parser = argparse.ArgumentParser(description='MFA')
    parser.add_argument('-f', '--file', metavar='file',
                        default='', help='Output from MFA')
    parser.add_argument('-c', '--clone', metavar='clone',
                        default='', help='Output from ClusterONE')
    return parser.parse_args()

def main(args):
    if (args.clone != ""):
        matB = readClusterOne(args.clone)
    matA = readBioPlexMFAOutput(args.file)
    
if __name__ == '__main__':
    main(get_args())    

    