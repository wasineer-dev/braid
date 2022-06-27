# braid
Find protein complexes from high-throughput AP-MS data

Python implementation for Markov Random Field model of AP-MS experiments.

Rungsarityotin, W., Krause, R., Schoedl, A., Schliep, A. (2007) BMC Bioinformatics 8:482.

#### Requirement
Require a Python >= 3.5 installation. Since it depends on some packages that can be tricky to install using pip (numba, numpy, ...), we recommend using the [Anaconda Python distribution](https://www.continuum.io/downloads). In case you are creating a new conda environment or using miniconda, please make sure to run `conda install anaconda` before running pip, or otherwise the required packages will not be present.  

In addition if you obtain the code from the development branch, you will need to install the package [statsmodel](https://www.statsmodels.org/stable/index.html)

#### Input file 
  (-f) a CSV file containing a list of bait-preys experiments. The first protein at the beginning of every line is a bait protein.
  The option (-bp) must be given if your input file is Bioplex 2.0 or Bioplex 3.0.
   
#### Model parameters
 1. The log-ratio of false negative rate and false positive rate (-psi)
 2. Maximum possible number of clusters (-k), e.g. 600 for gavin2002 dataset and 800 for gavin2006

#### Example

  `py main.py -f gavin2002.csv -psi 3.4 -k 500`
  or
  `py main.py -bp [Bioplex file] -psi 3.4 -k 500`
#### Output

 out.tab: a tab separating file containing a cluster annotation for each protein, one per line.

 out.sif: output clustering result in the Simple Interaction File format which you can import into Cytoscape.
