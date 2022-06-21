import argparse

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.cluster import pair_confusion_matrix, contingency_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import seaborn as sns; sns.set_theme()

setBenchmarkProteins = set()
setObservedProteins = set()

def readCYC2008():
    fileName = "CYC2008_complex.txt"
    clusters = {}
    proteins = {}
    with open(fileName) as fh:
        fh.readline() # skip header
        for line in fh:
            lst = line.rstrip().split('\t')
            # print(lst[0] + '\t' + lst[1])
            complex = lst[2]
            if complex not in clusters.keys():
                clusters[complex] = []
            clusters[complex].append(lst[0])
            setBenchmarkProteins.add(lst[0])
            proteins[lst[0]] = complex
        fh.close()
    print('CYC2008 ' + 'number of complexes = ' + str(len(clusters.keys())))
    return (proteins, clusters)

def readMFAOutput(fileName):
    clusters = {}
    proteins = {}
    with open(fileName) as fh:
        for line in fh:
            lst = line.rstrip().split('\t')
            # print(lst[0] + '\t' + lst[1])
            if lst[1] not in clusters.keys():
                clusters[lst[1]] = []
            clusters[lst[1]].append(lst[0])
            if lst[0] in setBenchmarkProteins and lst[0] not in proteins.keys():
                proteins[lst[0]] = lst[1]
        fh.close()
    print('MRF ' + 'number of complexes = ' + str(len(clusters.keys())))
    return (proteins, clusters)

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
            clusters[lst[1]].append(prot.strip())
            if prot in setBenchmarkProteins and prot not in proteins.keys():
                proteins[prot] = lst[1]
                setObservedProteins.add(prot.strip())
        fh.close()
    print("Number of observed proteins = ", len(setObservedProteins))
    print('MRF ' + 'number of complexes = ' + str(len(clusters.keys())))
    return (proteins, clusters)

def readEColi2018Benchmark():
    fileName = "ecoli_cep_localization.txt"
    clusters = {}
    proteins = {}
    with open(fileName) as fh:
        fh.readline() # skip header
        for line in fh:
            lst = line.rstrip().split('\t')
            complex = lst[0]
            lstProteins = []
            for prot in lst[2].strip("\"").split(','):
                lstProteins.append(prot.strip())
            #print(complex, '\t', lstProteins)
            if complex not in clusters.keys():
                clusters[complex] = lstProteins
            for prot in lstProteins:
                setBenchmarkProteins.add(prot.strip())
        fh.close()

    for k in clusters.keys():
        for prot in clusters[k]:
            if not prot in proteins.keys():
               proteins[prot] = list()
            proteins[prot].append(k)

    print("Number of benchmark proteins = ", len(setBenchmarkProteins))        
    return (proteins, clusters)

def filterProteins(y_actual, y_pred):
    z_actual = list()
    z_pred = list()
    for prot in setObservedProteins:
        for k in y_actual[prot]:
            z_actual.append(k)
            z_pred.append(y_pred[prot])
    return (z_actual, z_pred)

def sameComplex(setA, setB):
    return setA == setB

def assignClusterLabels(y_actual, y_pred):
    A = np.zeros(shape=(len(y_pred.keys()), len(y_actual.keys())), dtype=int)
    for i,k in zip(range(len(y_pred.keys())), y_pred.keys()):
        for j,m in zip(range(len(y_actual.keys())), y_actual.keys()):
            for prot in y_pred[k]:
                if prot in y_actual[m]:
                    A[i][j] += 1
    return A

def computeAllPairs(y_actual, y_pred, clusterA, clusterB):
    
    z_actual, z_pred = filterProteins(y_actual, y_pred)
    m = pair_confusion_matrix(z_actual, z_pred)
    print(m)
    SN = m[1][1]/float(m[1][1] + m[1][0])
    SP = m[1][1]/float(m[1][1] + m[0][1])
    print(SN, SP)
    
    cm = assignClusterLabels(clusterA, clusterB)
    ax = sns.heatmap(cm)
    plt.show()

def get_args():
    parser = argparse.ArgumentParser(description='MFA')
    parser.add_argument('-f', '--file', metavar='file',
                        default='', help='Output from MFA')
    return parser.parse_args()

def main(args):
    y_actual, clusterA = readEColi2018Benchmark()
    y_pred, clusterB = readEcoliMFAOutput(args.file)
    print("Number of proteins considered: ", len(y_pred.keys()))

    computeAllPairs(y_actual, y_pred, clusterA, clusterB)   

if __name__ == '__main__':
    main(get_args())    