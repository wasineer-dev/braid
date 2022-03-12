import mygene
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

mg = mygene.MyGeneInfo()

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

# S. cerevisiae, IntAct complex file (tsv format)
# Download on 12.03.2022
def readIntActComplex():
    df = pd.read_table("83333.tsv")
    nRows, nCols = df.shape
    clusters = {}
    for i in range(nRows):
        prots = set()
        for p in df.iloc[i][nCols-1].split('|'):
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
        gene_symbol = df.iloc[i][1].split("_")[0]
        uniprots[gene_symbol.lower()] = df.iloc[i][0]
    return uniprots

def readEcoliMFAOutput(fileName):
    uniprots = mapSymbol2Uniprot()
    clusters = {}
    proteins = {}
    with open(fileName) as fh:
        for line in fh:
            lst = line.rstrip().split('\t')
            prot = lst[0].split('__')[0]
            #print(prot + '\t' + lst[1])
            if lst[1] not in clusters.keys():
                clusters[lst[1]] = []
            prot = prot.strip().lower()
            if (prot in uniprots.keys()):
                clusters[lst[1]].append(uniprots[prot])
        fh.close()
    print('MRF ' + 'number of complexes = ' + str(len(clusters.keys())))

    predictions = {}
    for k in clusters.keys():
        for prot in clusters[k]:
            setObservedProteins.add(prot)
        if len(clusters[k]) > 1:
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
            if (prot.strip().lower() in uniprots.keys()):
                lstProteins.append(uniprots[prot.strip().lower()])
        # print(complex, '\t', lstProteins)
        if complex not in clusters.keys():
            clusters[complex] = lstProteins

    predictions = {}
    for k in clusters.keys():
        if len(clusters[k]) > 1:
            predictions[k] = clusters[k]
    return predictions

def measurement(matA, matB):
    matIndices = np.zeros((len(matA.keys()), len(matB.keys())), dtype='float')
    matCoverage = np.zeros((len(matA.keys()), len(matB.keys())), dtype='float')
    iA = list(matA.keys())
    vecPredictions = np.zeros(len(matA.keys()), dtype='float')
    for i, a in zip(range(len(matA.keys())), matA.keys()):
        setA = matA[a]
        for j, b in zip(range(len(matB.keys())), matB.keys()):
            matCoverage[i][j] = complexCoverage(setA, matB[b])
            nJaccard = jaccardIndex(setA, matB[b])
            matIndices[i][j] = nJaccard
        vecPredictions[i] = np.sum(matIndices[i,:] > 0.1)

    #
    # Calculate Predictions, Recalls, F-Measure
    # https://bmcmicrobiol.biomedcentral.com/articles/10.1186/s12866-020-01904-6
    #
    nPredictions = np.sum(vecPredictions > 0)/len(matA.keys())
    print('Prediction measure = ' + str(nPredictions))
    return matIndices

def get_args():
    parser = argparse.ArgumentParser(description='MFA')
    parser.add_argument('-f', '--file', metavar='file',
                        default='', help='Output from MFA')
    return parser.parse_args()

def main(args):
    matB = readIntActComplex()
    matA = readEcoliMFAOutput(args.file)
    matIndices = measurement(matA, matB)
    
    sns.heatmap(matIndices)
    plt.show()

    matC = readEcoliBabu2018()
    matIndices = measurement(matC, matB)
    sns.heatmap(matIndices)
    plt.show()

if __name__ == '__main__':
    main(get_args())    

    