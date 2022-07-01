import mygene
import argparse

mg = mygene.MyGeneInfo()
geneDict = dict()

def convertInteractionFile(fileName):
    
    xli = []
    with open(fileName) as fh:
        for line in fh:
            lst = line.rstrip().split('\t')
            protA = lst[0]
            t = lst[1]
            protB = lst[2]
            if (not protA in geneDict.keys()):
                geneDict[protA] = protA
                xli.append(protA)
            if (not protB in geneDict.keys()):
                geneDict[protB] = protB
                xli.append(protB)
        fh.close()
    print(str(len(xli)))
    if (len(xli) > 0):
        out = mg.querymany(xli, scopes='symbol,accession,ensembl.gene', fields='symbol, entrezgene')
        for i in range(len(xli)):
            if ('symbol' in out[i].keys()):
                geneDict[out[i]['query']] = out[i]['symbol']

    with open(fileName) as fh:
        with open('convert2GeneSymbol.sif', 'w') as fhout:
            for line in fh:
                lst = line.rstrip().split('\t')
                protA = lst[0]
                if (protA in geneDict.keys()):
                    protA = geneDict[protA]
                t = lst[1]
                protB = lst[2]
                if (protB in geneDict.keys()):
                    protB = geneDict[protB]
                fhout.write(protA + '\t' + t + '\t' + protB + '\n')
            fhout.close()
        fh.close()

def get_args():
    parser = argparse.ArgumentParser(description='SIF')
    parser.add_argument('-f', '--file', metavar='file',
                        default='', help='SIF file, output from MFA')
    return parser.parse_args()

def main(args):
    convertInteractionFile(args.file)

if __name__ == '__main__':
    main(get_args())