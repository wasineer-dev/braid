
##
## https://stats.stackexchange.com/questions/396315/coding-a-simple-stick-breaking-process-in-python
##

import numpy as np

# A nice correction suggested by Tomáš Tunys
def Stick_Breaking(num_weights,alpha):
    betas = np.random.beta(1,alpha, size=num_weights)
    betas[1:] *= np.cumprod(1 - betas[:-1])       
    return betas


import matplotlib.pyplot as plt

for _ in range(5):
    num_weights = 2
    alpha = 100
    weights = Stick_Breaking(num_weights,alpha)
    plt.axis([0, num_weights+1, 0, max(weights)])
    plt.bar(range(1,num_weights+1),weights)
    plt.show()
    print(sum(weights))