# braid
Find protein complexes from high-throughput AP-MS data

Python implementation for Markov Random Field model of AP-MS experiments.

Rungsarityotin, W., Krause, R., Schoedl, A., Schliep, A. (2007) BMC Bioinformatics 8:482.

Installation requirement: numpy, scipy and matplotlib

Input file: (-f) a CSV file containing a list of bait-preys experiments
The first protein at the beginning of every line is a bait protein.
   
Model parameters:
 1. The log-ratio of false negative rate and false positive rate (-psi)
 2. Maximum possible number of clusters (-k), e.g. 600 for gavin2002 dataset and 700 for gavin2006

Example:
py main.py -f gavin2002.csv -psi 3.4 -k 500
