# braid
Find protein complexes from high-throughput AP-MS data

Python implementation for Markov Random Field model of AP-MS experiments.

Rungsarityotin, W. et al. (2007) BMC Bioinformatics 8:482.

Installation requirement: numpy, scipy and matplotlib

Input file: a CSV file containing a list of bait-preys experiments
The first protein at the beginning of every line is a bait protein.
   
Model parameters:
 1. The log-ratio of false negative rate and false positive rate (-psi)
 2. Maximum possible number of clusters (-k), e.g. 600 for gavin2002 dataset and 700 for gavin2006
