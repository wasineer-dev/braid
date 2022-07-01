
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

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
        fh.close()
    print('CYC2008 ' + 'number of complexes = ' + str(len(clusters.keys())))
    return clusters

def readMCLOutput():
    fileName = "Annotated_YHTP2008_complex.txt"
    clusters = {}
    proteins = {}
    nK = 0
    with open(fileName) as fh:
        fh.readline() # skip header
        for line in fh:
            lst = line.rstrip().split('\t')
            if len(lst) < 2:
                break
            if lst[0] != '':
                nK = int(lst[0])
                clusters[nK] = []
                # print(lst[0] + '\t' + lst[1])
            else:
                clusters[nK].append(lst[1])
    print('MCL ' + 'number of complexes = ' + str(len(clusters.keys())))
    return clusters

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
    return clusters

def readEcoliMFAOutput(fileName):
    clusters = {}
    proteins = {}
    with open(fileName) as fh:
        for line in fh:
            lst = line.rstrip().split('\t')
            prot = lst[0].split('__')[0]
            #print(prot + '\t' + lst[1])
            if lst[1] not in clusters.keys():
                clusters[lst[1]] = []
            clusters[lst[1]].append(prot.strip())
            setObservedProteins.add(prot.strip())
        fh.close()
    print('MRF ' + 'number of complexes = ' + str(len(clusters.keys())))
    return clusters

def readEColi2018Benchmark():
    fileName = "ecoli_bait_prey_complexes.txt"
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
            if prot not in proteins.keys():
                proteins[prot] = set()
            proteins[prot].add(k)

    print("Number of benchmark proteins = ", len(setBenchmarkProteins))        
    return (proteins, clusters)

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
    print('Measure = ' + str(nPredictions))
    return nPredictions

def sortBySize(matA):
    vecKeys = list(matA.keys())
    vecSize = []
    for k in vecKeys:
        vecSize.append(len(matA[k]))
    indices = np.argsort(vecSize)
    sortedKeys = []
    for i in indices:
        sortedKeys.append(vecKeys[i])
    return sortedKeys

def main(args):
    # readMCLOutput()
    matA = readEcoliMFAOutput(args.file)
    # matB = readCYC2008()
    y_actual, matB = readEColi2018Benchmark()

    print("Number of proteins considered = ", len(setBenchmarkProteins.intersection(setObservedProteins)))

    matIndices = np.zeros((len(matA.keys()), len(matB.keys())), dtype='float')
    matCoverage = np.zeros((len(matA.keys()), len(matB.keys())), dtype='float')
    iA = list(matA.keys())
    lstScores = []
    vecPredictions = np.zeros(len(matA.keys()), dtype='float')
    for i, a in zip(range(len(matA.keys())), sortBySize(matA)):
        setA = matA[a]
        for j, b in zip(range(len(matB.keys())), sortBySize(matB)):
            matCoverage[i][j] = complexCoverage(setA, matB[b])
            nJaccard = jaccardIndex(setA, matB[b])
            matIndices[i][j] = nJaccard
            if nJaccard > 0.1:
                lstScores.append(nJaccard)
        vecPredictions[i] = np.sum(matIndices[i,:] > 0.1)

    #
    # Calculate Predictions, Recalls, F-Measure
    # https://bmcmicrobiol.biomedcentral.com/articles/10.1186/s12866-020-01904-6
    #
    nPredictions = np.sum(vecPredictions > 0)/len(matA.keys())
    print('Prediction measure = ' + str(nPredictions))

    nRecalls = measurement(matB, matA)
    nFMeasure = 2* (nPredictions * nRecalls)/(nPredictions + nRecalls)
    print('F-Measure = ' + str(nFMeasure))

    matMatches = []
    nRows,nCols = matIndices.shape
    for i in range(nRows):
        matMatches.append(100.*matIndices[i,:])

    lstSizes = []
    singletons = 0
    for k in matA.keys():
        if (len(matA[k]) > 1):
            lstSizes.append(len(matA[k]))
        else:
            singletons += 1
    maxSize = np.max(lstSizes)

    ## Output ordered by cluster
    if False:
        with open("gavin2006_clusters.tab", "w") as fh:
            fh.write("Cluster\tORF\n")
            for k in sortedKeys:
                strLine = k + '\t' + matA[k][0]
                if len(matA[k]) > 1:
                    if len(matA[k]) > 10:
                        print('Cluster ' + k + ', containing ' + str(len(matA[k])) + ' members')
                    for strName in matA[k][1:]:
                        strLine += (',' + strName)
                if len(matA[k]) > 2:
                    print(strLine)
                strLine += '\n'
                if (len(matA[k]) > 2):
                    fh.write(strLine)
            fh.close()
    
    print('MRF largest cluster = ' + str(maxSize))
    plt.hist(lstSizes, range(200))
    plt.title('Complex sizes in Gavin2002')
    plt.show()

    sns.heatmap(matCoverage)
    plt.show()

if __name__ == '__main__':
    main(get_args())