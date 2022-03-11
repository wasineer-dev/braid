import argparse

import numpy as np
import matplotlib.pyplot as plt

setBenchmarkProteins = set()

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


def computeAllPairs(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    pos = 0
    for i in y_pred.keys():
        for j in y_pred.keys():
            if i == j:
                continue

            if y_actual[i] == y_actual[j]:
                pos += 1

            if y_actual[i] == y_actual[j] and y_pred[i] == y_pred[j]:
                TP += 1

            if y_actual[i] == y_actual[j] and y_pred[i] != y_pred[j]:
                FN += 1
            
            if y_actual[i] != y_actual[j] and y_pred[i] == y_pred[j]:
                FP += 1
            
            if y_actual[i] != y_actual[j] and y_pred[i] != y_pred[j]:
                TN += 1

    sensitivity = float(TP)/float(pos)
    specificity = float(TP)/float(TP + FP)
    return (sensitivity, specificity)

def get_args():
    parser = argparse.ArgumentParser(description='MFA')
    parser.add_argument('-f', '--file', metavar='file',
                        default='', help='Output from MFA')
    return parser.parse_args()

def main(args):
    y_actual, matB = readCYC2008()
    y_pred, matA = readMFAOutput(args.file)
    print("Number of proteins considered: ", len(y_pred.keys()))

    SN, SP = computeAllPairs(y_actual, y_pred)   

    print("Sensitivity = ", SN)
    print("Specificity = ", SP)

if __name__ == '__main__':
    main(get_args())    