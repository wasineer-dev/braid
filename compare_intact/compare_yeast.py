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
    df = pd.read_table("559292.tsv")
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

# convert our protein ORF to uniprot identifier
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
        fh.close()
    print('MRF ' + 'number of complexes = ' + str(len(clusters.keys())))

    predictions = {}
    for k in clusters.keys():
        uniprots = set()
        xli = clusters[k]
        if len(xli) < 2:
            continue
        out = mg.querymany(xli, scopes='symbol,accession,ensembl.gene', fields='symbol, entrezgene, uniprot')
        for i in range(len(xli)):
            if ('uniprot' in out[i].keys()):
                print(out[i]['uniprot'])
                if isinstance(out[i]['uniprot']['Swiss-Prot'], list):
                    for p in set(out[i]['uniprot']['Swiss-Prot']):
                        uniprots.add(p)
                else:
                    uniprots.add(out[i]['uniprot']['Swiss-Prot'])    
        predictions[k] = uniprots
        print(uniprots)
    return predictions

def get_args():
    parser = argparse.ArgumentParser(description='MFA')
    parser.add_argument('-f', '--file', metavar='file',
                        default='', help='Output from MFA')
    return parser.parse_args()

def main(args):
    matB = readIntActComplex()
    matA = readMFAOutput(args.file)
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

    sns.heatmap(matIndices)
    plt.show()

if __name__ == '__main__':
    main(get_args())    

    