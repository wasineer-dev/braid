
import random
import countSpokeModel

class CSpokeModel:

    def __init__(self, nNode, nCluster, fp, fn):
        self.nNode = nNode
        self.nCluster = nCluster
        self.fp = fp
        self.fn = fn

        self.listLabels = []
        self.listBaits = []

        for i in range(0, nNode):
            self.listLabels.append(random.randrange(0, nCluster))

    def generate(self, nTrials):
        
        bait = random.randrange(0, self.nNode)
        self.listLabels.append(bait)
        bait_label = self.listLabels[bait]

        observed = [bait]

        for i in range(0, nTrials):
            prey = random.randrange(0, self.nNode)

            if (bait == prey):
                continue

            # Choose success or failed trial
            if bait_label == self.listLabels[prey]:
                if random.random() < (1-self.fn): # true positive
                    observed.append(prey)
            else:
                if random.random() < self.fp: # false positive
                    observed.append(prey)
        print(observed)
        return observed

def main():
    spokeModel = CSpokeModel(500, 12, 0.4, 0.001)
    print(spokeModel.listLabels)

    listIndices = []
    for i in range(0, 600):
        listIndices.append(spokeModel.generate(10))

    countSpokeModel.CountSpokeModel(500, spokeModel.listBaits, listIndices)

if __name__ == '__main__':
    main()



