#!/usr/bin/env python3

#
# BRAID: A python program to find protein complexes from high-throughput data
#
#

import argparse

import countSpokeModel as cpm

def read_input(filename):
    with open(filename) as fh:
        records = dict()
        listInput = []
        for line in fh:
            lst = line.rstrip().split(',')
            print(lst)
            listInput.append(lst)
            for protein in lst:
                records[protein] = 0
        states = list(records.keys())
        sorted(states)
        print('Number of proteins ' + str(len(states)))

    nProteins = len(states)
    listBaits = []
    for lst in listInput:
        bait = lst[0]
        listBaits.append(states.index(bait))
    print('Number of purifications ' + str(len(listBaits)))

    listIndices = []
    for lst in listInput:
        indices = []
        for prot in lst:
            indices.append(states.index(prot))
        listIndices.append(indices)

    cpm.CountSpokeModel(nProteins, listBaits, listIndices)

def get_args():
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('-f', '--file', metavar='file',
                        default='', help='CSV input file of protein purifications')
    return parser.parse_args()

def main():
    args = get_args()
    print('Hello, ' + args.file + '!')
    read_input(args.file)

if __name__ == '__main__':
    main()