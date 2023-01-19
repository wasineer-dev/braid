import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics.cluster import contingency_matrix, pair_confusion_matrix, adjusted_rand_score

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

uniprots = mapSymbol2Uniprot()
setBenchmarkProteins = set()
setObservedProteins = set()

def query_uniprots(id):
    for gene in uniprots.keys():
        if id == uniprots[gene]:
            return id 
    return None

def jaccardIndex(vecA, vecB):
    setProteins = setBenchmarkProteins
    setA = set()
    for prot in vecA:
        setA.add(prot)
    setB = set()
    for prot in vecB:
        setB.add(prot)
    nIntersect = setA.intersection(setB)
    return float(len(nIntersect))/len(setA.union(setB))
    
def complexCoverage(vecA, vecB):
    setProteins = setBenchmarkProteins
    setA = set()
    for prot in vecA:
        setA.add(prot)
    setB = set()
    for prot in vecB:
        setB.add(prot)
    nIntersect = setA.intersection(setB)
    return float(len(nIntersect))/float(len(setA))
    
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
            prot = p.split('(')[0]
            if (query_uniprots(prot) != None):
                prots.add(prot)    
        print(df.iloc[i][0], prots)
        clusters[df.iloc[i][0]] = prots
    for k in clusters.keys():
        for prot in clusters[k]:
            setBenchmarkProteins.add(prot)
    
    with open("ecoli_intact_complexes.txt", "w") as fh:
        for k, cl in enumerate(clusters):
            for prot in clusters[cl]:
                fh.write(prot + '\t')
            fh.write('\n')
        fh.close()
    return clusters

def readEcoliMFAOutput(fileName):
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

    with open("ecoli_mrf_complexes.txt", "w") as fh:
        for k, cl in enumerate(clusters):
            for prot in clusters[cl]:
                fh.write(prot + '\t')
            fh.write('\n')
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
            if (prot in uniprots.keys()):
                lstProteins.append(uniprots[prot.strip()])
        if complex not in clusters.keys():
            clusters[complex] = lstProteins
            # print(complex, '\t', lstProteins)

    with open("ecoli_babu18_complexes.txt", "w") as fh:
        for k, cl in enumerate(clusters):
            for prot in clusters[cl]:
                fh.write(prot + '\t')
            fh.write('\n')
        fh.close()

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

def measurement(matA, matB):
    matIndices = np.zeros((len(matA.keys()), len(matB.keys())), dtype='float')
    iA = list(matA.keys())
    vecPredictions = np.zeros(len(matA.keys()), dtype='float')
    nClusters = sum(list( len(matA[i]) > 0 for i in matA.keys()) )
    for i in range(len(matA.keys())):
        setA = matA[iA[i]]
        if (len(setA) > 1):
            iB = list(matB.keys())
            for j in range(len(matB.keys())):
                nJaccard = jaccardIndex(setA, matB[iB[j]])
                matIndices[i][j] = nJaccard
            vecPredictions[i] = np.sum(matIndices[i,:] > 0.1)
    nPredictions = np.sum(vecPredictions > 0)/nClusters
    return nPredictions

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

    nRecalls = measurement(matB, matA)
    print('Recalls = ' + str(nRecalls))
    
    nFMeasure = 2* (nPredictions * nRecalls)/(nPredictions + nRecalls)
    print('F-Measure = ' + str(nFMeasure))

if __name__ == '__main__':
    main(get_args())    

    