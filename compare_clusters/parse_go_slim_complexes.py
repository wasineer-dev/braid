#!/usr/bin/env python3

import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argsort

def parse_GO(str):
    lst = str.split('/')
    if len(lst) < 3:
        return ''
    else:
        return lst[0] + '/' + lst[1] + '/' + lst[2]

def read_go_slim_complexes():
    fileName = '../curation_yeast/go_protein_complex_slim.tab'
    clusterKeys = {}
    with open(fileName) as fh:
        for line in fh:
            lst = line.rstrip().split('\t')
            strGOName = lst[0]
            clusterKeys[strGOName] = []
            lstNames = lst[1].split('|')
            strNames = parse_GO(lstNames[0])
            clusterKeys[strGOName].append(parse_GO(lstNames[0]))
            for str in lstNames[1:]:
                lst = str.split('/')
                if len(lst) >= 3:
                    strNames += '|' + parse_GO(str)
                    clusterKeys[strGOName].append(parse_GO(str))
            ## print(strGOName + ' ' + strNames)
    return clusterKeys

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

def findProteinName(strName, dictGOSlim):
    for goterm in dictGOSlim.keys():
        for strA in dictGOSlim[goterm]:
            if strA.find(strName) >= 0:
                return goterm
    return ''

def findGOTerms():
    dictGOSlim = read_go_slim_complexes()
    dictMFA = readMFAOutput('../MFAClusters/out_gavin2006.tab')

    dictGOTerms = {}
    for cluster in dictMFA.keys():
        lstProteins = dictMFA[cluster]
        setTerms = set()
        for s in lstProteins:
            goterm = findProteinName(s, dictGOSlim)
            setTerms.add(goterm)
        dictGOTerms[cluster] = setTerms

    for cluster in dictGOTerms.keys():
        line = cluster + '\t'
        for goname in dictGOTerms[cluster]:
            line += ' ' + goname

    # Counting how many GO Terms are split into different clusters
    dictGOHist = {}
    for cluster in dictGOSlim.keys():
        dictGOHist[cluster] = set()
    for cluster in dictMFA.keys():
        lstProteins = dictMFA[cluster]
        for s in lstProteins:
            goterm = findProteinName(s, dictGOSlim)
            if goterm != '':
                dictGOHist[goterm].add(cluster)

    # Plot GO Terms representation
    vecGOTerms = list(dictGOSlim.keys())
    vecTermCounts = []
    for k in vecGOTerms:
        vecTermCounts.append(len(dictGOHist[k]))
    
    with open('goterms_MFA_Gavin2006.csv', 'w') as fh:
        vecArgSorted = argsort(vecTermCounts)
        for i in vecArgSorted:
            if (vecTermCounts[i] > 0):
                fh.write(vecGOTerms[i] + ',' + str(vecTermCounts[i]))
                fh.write('\n')
        fh.close()

if __name__ == '__main__':
    findGOTerms()    