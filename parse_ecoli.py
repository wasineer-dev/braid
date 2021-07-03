#!/usr/bin/env python3
#
#
#
# Transform raw experimental data from Babu et al. (E. coli) into a set of purifications
# 
#

import argparse

def read_input(filename):
    with open(filename) as fh:
        for line in fh:
            lst = line.rstrip()
            print(line)


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