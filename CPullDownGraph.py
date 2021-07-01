#
# CPullDownGraph
#

class CNode:
    
    def __self__(self, nodeV, nFrequency):
        self.nodeV = nodeV
        self.nFrequency = nFrequency

class CEdge:

    def __self__(self, nodeV, nTrial, nSuccess):
        self.nodeV = nodeV
        self.nTrial = nTrial
        self.nSuccess = nSuccess

class CPullDownGraph:

    def __self__(self, nProteins):
        self.nProteins = nProteins
