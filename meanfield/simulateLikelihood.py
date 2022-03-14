from unittest import expectedFailure
import numpy as np
from numpy.random import default_rng

import matplotlib.pyplot as plt
import sys
import scipy.special
import scipy.stats

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.pipeline import Pipeline

import statsmodels.api as sm
from scipy import stats

# A nice correction suggested by Tomáš Tunys
def Stick_Breaking(num_weights,alpha):
    betas = np.random.beta(1,alpha, size=num_weights)
    betas[1:] *= np.cumprod(1 - betas[:-1])  
    betas /= np.sum(betas)     
    return betas

# betas is a multinomial distribution
def Assign_Cluster(rng, betas):
    t = rng.random()
    sorted_arg = np.argsort(betas)
    indicator = np.zeros(len(betas))
    cumulativeSum = 0.0
    for i in range(len(betas)):
        if t >= cumulativeSum and t < (cumulativeSum + betas[sorted_arg[i]]):
            indicator[sorted_arg[i]] = 1
            return indicator
        cumulativeSum += betas[sorted_arg[i]]
    assert(True)
#
# Observation graph
# 
def ObservationGraph(Nproteins, Nk, fn, fp, alpha):
    rng = default_rng()
    NTrials = 5
    #
    # Try stick breaking weights
    # 
    alphas = np.ones(Nk)*alpha
    betas = np.random.dirichlet(alphas) # Sample from the Dirichlet distribution
    sizeDistribution = np.random.multinomial(Nproteins, betas, 1)
    lstDistribution = []
    for p in sizeDistribution[0]:
        lstDistribution.append(p/Nproteins)
    mIndicatorQ = np.zeros((Nproteins, Nk), dtype=float)
    for i in range(Nproteins):
        mIndicatorQ[i,:] = Assign_Cluster(rng, lstDistribution)

    print("Cluster = " + str(mIndicatorQ))
    mObservationSuccess = np.zeros((Nproteins, Nproteins), dtype=int)

    for i in range(Nproteins):
        for j in range(i):
            if (np.argmax(mIndicatorQ[i]) == np.argmax(mIndicatorQ[j])):
                mObservationSuccess[i][j] += np.random.binomial(NTrials, 1-fn, 1)
            else:
                mObservationSuccess[i][j] += np.random.binomial(NTrials, fp, 1)

    return mObservationSuccess

def tail_bound(k, n, p):
    a = float(k)/float(n)
    return np.exp(-2.0*n*np.power(p - a, 2))

def chernoff_bound(k, n, p):    
    if n == 0:
        return 1.0
        
    a = float(k)/float(n)
    if a == 0:
        kl = -np.log(1.0 - p)
    else:
        kl = a*(np.log(a) - np.log(p)) + (1.0 - a)*(np.log(1.0 - a) - np.log(1.0 - p))
    return np.exp(-n*kl)

class CMeanFieldAnnealing:

    def __init__(self, Nproteins, Nk):
        self.lstExpectedLikelihood = []
        self.mIndicatorQ = np.zeros((Nproteins, Nk), dtype=float)
        
    def Likelihood(self, mObservationG, Nproteins, Nk, psi):

        rng = default_rng()

        # psi = (-np.log(fp) + np.log(1 - fn))/(-np.log(fn) + np.log(1 - fp))
        print('psi = ', psi)

        for i in range(Nproteins):
            self.mIndicatorQ[i,:] = rng.random(Nk)
            self.mIndicatorQ[i,:] /= sum(self.mIndicatorQ[i,:])

        nTemperature = 1000.0
        # TODO: refactor 
        while nTemperature >= 1.0:
            nLastLogLikelihood = 0.0
            nIteration = 0
            for i in range(Nproteins):
                # i = np.random.randint(0, Nproteins) # Choose a node at random
                """ 
                mLogLikelihood = np.zeros(Nk, dtype=float) # Negative log-likelihood
                for k in range(Nk):
                    for j in mObservationG.lstAdjacency[i]:
                        t = mObservationG.mTrials[i][j]
                        s = mObservationG.mObserved[i][j]
                        assert(s <= t)
                        mLogLikelihood[k] += (self.mIndicatorQ[j][k]*float(t-s) + (1.0 - self.mIndicatorQ[j][k])*float(s)*psi)
                        ## mLogLikelihood[k] += self.mIndicatorQ[j][k]*(t-s-s*psi)
                """

                fn_out = np.matmul(mObservationG.mTrials[i] - mObservationG.mObserved[i], self.mIndicatorQ) 
                fp_out = np.matmul(psi*mObservationG.mObserved[i], np.ones((Nproteins, Nk)) - self.mIndicatorQ)

                mLogLikelihood = fn_out + fp_out

                # Overflow problem. Need to compute with softmax
                gamma = nTemperature
                self.mIndicatorQ[i,:] = scipy.special.softmax(-gamma*mLogLikelihood)
                ## self.mIndicatorQ[i,:] /= sum(self.mIndicatorQ[i,:])
                
            nIteration += 1

            nTemperature *= 0.5

        return self.lstExpectedLikelihood

    ##
    ## Adapt from https://github.com/zib-cmd/cmdtools/blob/dev/src/cmdtools/analysis/optimization.py
    ##
    def indexsearch(self, X):
        """ Return the indices to the rows spanning the largest simplex """

        n = np.size(X, axis=0)
        k = np.size(X, axis=1)
        X = X.copy()

        ind = np.zeros(k, dtype=int)
        for j in range(0, k):
            # find the largest row
            rownorm = np.linalg.norm(X, axis=1)
            ind[j] = np.argmax(rownorm)

            if j == 0:
                # translate to origin
                X -= X[ind[j], :]
            else:
                # remove subspace of this row
                if (rownorm[ind[j]] != 0.0):
                    X /= rownorm[ind[j]]
                v  = X[ind[j], :]
                X -= np.outer(X.dot(v), v)

        return ind

    def computeErrorRate(self, mObservationG, Nproteins):
        self.indicatorVec = np.argmax(self.mIndicatorQ, axis=1)

        rnk = np.linalg.matrix_rank(self.mIndicatorQ)
        print("Indicator matrix had rank = " + str(rnk))
        nClusters = len(np.unique(self.indicatorVec))
        print("Number of clusters used: " + str(nClusters))

        countFn = 0
        countFp = 0
        sumSameCluster = 0
        sumDiffCluster = 0
        for i in range(Nproteins):
            for j in mObservationG.lstAdjacency[i]:
                t = mObservationG.mTrials[i][j]
                s = mObservationG.mObserved[i][j]
                assert(s <= t)
                if (self.indicatorVec[i] == self.indicatorVec[j]):
                    countFn += (t - s)
                    sumSameCluster += t
                else:
                    countFp += s
                    sumDiffCluster += t

        fn = 0.0
        fp = 0.0
        if (sumSameCluster > 0):
            fn = float(countFn)/float(sumSameCluster)
        if (sumDiffCluster > 0):
            fp = float(countFp)/float(sumDiffCluster)
        return (fn, fp)

    def estimator_summary(self, regr, y_actual, y_pred):
        # The coefficients
        print("Coefficients: \n", regr.coef_)
        # The mean squared error
        print("Mean squared error: %.2f" % mean_squared_error(y_actual, y_pred))
        # The coefficient of determination: 1 is perfect prediction
        print("Coefficient of determination: %.2f" % r2_score(y_actual, y_pred))
        
    def computeResidues(self, mObservationG, Nproteins, Nk):
    
        (fn, fp) = self.computeErrorRate(mObservationG, Nproteins)

        # Filter singletons
        clusters = []
        singletons = []
        for k in range(Nk): 
            if (sum(self.indicatorVec == k) > 1):
                clusters.append(k)
            else:
                singletons.append(k)
        setProteins = set()
        indicators = dict()
        for i in range(Nproteins):
            if self.indicatorVec[i] in clusters: 
                setProteins.add(i)
                indicators[i] = self.indicatorVec[i]

        countFn = np.zeros(Nk)
        countFp = np.zeros(Nk)
        trialFn = np.zeros(Nk)
        trialFp = np.zeros(Nk)
        for i in setProteins:
            cl = indicators[i]    
            for j in mObservationG.lstAdjacency[i]:
                t = mObservationG.mTrials[i][j]
                s = mObservationG.mObserved[i][j]
                if (self.indicatorVec[i] == cl) and (self.indicatorVec[j] == cl):      
                    trialFn[cl] += t
                    countFn[cl] += (t - s)
                if (self.indicatorVec[i] == cl) and (self.indicatorVec[j] != cl):
                    countFp[cl] += s
                    trialFp[cl] += t

        self.expectedErrors = np.floor(fn*trialFn) + np.floor(fp*trialFp)
        self.mResidues = countFn + countFp

        expectedErrors = list( self.expectedErrors[i] for i in clusters)
        residues = list( self.mResidues[i] for i in clusters)
        result = scipy.stats.ks_2samp(residues, expectedErrors)
        print(result)

        result = scipy.stats.cramervonmises_2samp(residues, expectedErrors)
        print(result)

        self.expectedErrors = expectedErrors
        self.mResidues = residues

        # Ordinary Least Square
        X = np.reshape(expectedErrors, (-1,1))
        est = sm.OLS(residues,  X)
        est2 = est.fit()
        print(est2.summary())

        glm_poisson = sm.GLM(residues, X, family=sm.families.Poisson(sm.families.links.log()))
        glm_results = glm_poisson.fit()
        print(glm_results.summary())

        # Create linear regression object
        regr = linear_model.LinearRegression()
        # Train the model using the training sets
        regr.fit(X, residues)
        # Make predictions using the testing set
        y_pred = regr.predict(X)
        print("Linear Regression evaluation:")
        self.estimator_summary(regr, residues, y_pred)
        
        return (est2, fn, fp)

    def computeEntropy(self, Nproteins, Nk):
        self.mEntropy = np.zeros(Nproteins, dtype=float)
        for i in range(Nproteins):
            for k in range(Nk):
                p = self.mIndicatorQ[i][k]
                if (p > 0):
                    self.mEntropy[i] += (-p*np.log(p))
        
    #
    # https://www.researchgate.net/publication/343831904_NumPy_SciPy_Recipes_for_Data_Science_Projections_onto_the_Standard_Simplex
    #
    def indicator2simplex(self):
        m,n = self.mIndicatorQ.shape
        matS = np.sort(self.mIndicatorQ, axis=0)
        matC = np.cumsum(matS, axis=0) - 1.
        matH = matS - matC / (np.arange(m) + 1).reshape(m,1)
        matH[matH<=0] = np.inf

        r = np.argmin(matH, axis=0)
        t = matC[r,np.arange(n)] / (r + 1)

        matY = self.mIndicatorQ - t
        matY[matY < 0] = 0
        return matY

    def clusterImage(self, matQ):
        m,n = matQ.shape
        vecArgMax = np.argmax(matQ,axis=1)
        vecIndices = np.argsort(vecArgMax)
        matImage = np.zeros((m,n), dtype=int)
        for i in range(len(vecIndices)):
            k = vecArgMax[vecIndices[i]]
            matImage[i][k] = 255
        return matImage

if __name__ == '__main__':
    NPROTEINS = 100
    NCLUSTERS = 10
    mGraph = ObservationGraph(NPROTEINS, NCLUSTERS, 0.001, 0.01, 10)

    lstCostFunction = []
    fn = 0.001
    fp = 0.01
    for k in range(2,50):
        cmfa = CMeanFieldAnnealing(NPROTEINS, NCLUSTERS)
        minCost = cmfa.Likelihood(mGraph, NPROTEINS, k, 3.0)
        lstCostFunction.append(minCost)

    plt.plot(range(2,50), lstCostFunction)
    plt.show()

#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.plot3D(mA[:,0], mA[:,1], mA[:,2])
#plt.show()
