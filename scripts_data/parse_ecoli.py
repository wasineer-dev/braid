#!/usr/bin/env python3
#
#
#
# Transform raw experimental data from Babu et al. (E. coli) into a set of purifications
# 
# Input file format:
# skip the header
# We are only interested in
# 1st Column: index to run
# 3rd Column: bait of the run
# 4rd Column: prey observed to interact with the bait in this run
#
#
# Output file format:
# Each line contains an AP-MS purification experiment
# bait, prey1, prey2, ...
#

import argparse

def parse_input(filename):
    records = dict()
    
    with open(filename) as fh:
        fh.readline() # skip the header     
        for line in fh:
            lst = line.rstrip().split()
            bait = lst[2] 
            prey = lst[3]
            if not lst[0] in records.keys():
                records[lst[0]] = dict()

            if not bait in records[lst[0]].keys():
                records[lst[0]][bait] = []

            records[lst[0]][bait].append(prey)

    proteins = dict()
    with open('ecoli_nature2018_apms.txt', 'w') as fhOut:
        for i in records.keys():
            for bait in records[i].keys():
                pur = bait
                if not bait in proteins.keys():
                    proteins[bait] = 0
                proteins[bait] += 1
                for prey in records[i][bait]:
                    pur += ',' + prey
                    if not prey in proteins.keys():
                        proteins[prey] = 0
                    proteins[prey] += 1
                print(pur, file=fhOut)
        fhOut.close()

    print('Total number of proteins ' + str(len(proteins.keys())))
    print('Total number of purifications ' + str(len(records.keys())))

def get_args():
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('-f', '--file', metavar='file',
                        default='', help='CSV input file of protein purifications')
    return parser.parse_args()

def main():
    args = get_args()
    print('Hello, ' + args.file + '!')
    parse_input(args.file)

if __name__ == '__main__':
    main()