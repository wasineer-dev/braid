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

from scipy import stats

import tensorflow as tf

MAX_ITERATION = 100
        
class CMeanFieldAnnealing:

    def __init__(self, Nproteins, Nk):
        self.lstExpectedLikelihood = []
        self.mIndicatorQ = np.zeros((Nproteins, Nk), dtype=float)
        self.alpha = 0.1
        self.beta = 0.1

    def annealing(self, mix_p, mObservationG, Nproteins, Nk, psi):

        matA = np.array(mObservationG.mTrials - mObservationG.mObserved, dtype=float)
        matB = np.array(psi*mObservationG.mObserved, dtype=float)

        gamma = 1000.0
        nIteration = 0
        prev = np.finfo(np.float32).max
        while(nIteration < MAX_ITERATION):
            for i in range(Nproteins):        
                fn_out = np.tensordot(mObservationG.mTrials[i] - mObservationG.mObserved[i], self.mIndicatorQ, axes=1) 
                fp_out = np.tensordot(psi*mObservationG.mObserved[i], 1.0 - self.mIndicatorQ, axes=1)

                mLogLikelihood = fn_out + fp_out
                self.mIndicatorQ[i,:] = scipy.special.softmax(-gamma*mLogLikelihood)
            nIteration += 1
            
            mLogLikelihood = 0.0
            for i in range(Nproteins):        
                fn_out = np.tensordot(mObservationG.mTrials[i] - mObservationG.mObserved[i], self.mIndicatorQ, axes=1) 
                fp_out = np.tensordot(psi*mObservationG.mObserved[i], 1.0 - self.mIndicatorQ, axes=1)
                mLogLikelihood += np.sum(fn_out + fp_out)
            print("MFA: num. iterations = ", nIteration, mLogLikelihood)
        
            (fn, fp, _) = self.computeErrorRate(psi, mObservationG, Nproteins)
            psi = self.compute_psi(fp, fn)
            print("FN: fn rate = ", fn)
            if np.isclose(mLogLikelihood, prev, rtol=0.01):
                return mLogLikelihood
            else:
                prev = mLogLikelihood
        return mLogLikelihood

    def tf_annealing(self, mix_p, mObservationG, Nproteins, Nk, psi):

        matA = tf.convert_to_tensor(mObservationG.mTrials - mObservationG.mObserved, dtype=tf.float32)
        matB = tf.convert_to_tensor(psi*mObservationG.mObserved, dtype=tf.float32)
        tfArray = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        for i in range(Nproteins):
            tfArray = tfArray.write(i, self.mIndicatorQ[i])
        tQ = tfArray.stack()
        gamma = 10000.0
        nIteration = 0
        prev = np.finfo(np.float32).max
        while(gamma > 1.0 and nIteration < MAX_ITERATION):
            for i in range(Nproteins):
                fn_out = tf.tensordot(matA[i], tQ, axes=1) 
                fp_out = tf.tensordot(matB[i], 1.0 - tQ, axes=1)
                mLogLikelihood = fn_out + fp_out
                indices = [(i,j) for j in range(Nk)]
                tQ = tf.tensor_scatter_nd_update(tQ, indices, tf.nn.softmax(-gamma*mLogLikelihood))
            
            nIteration += 1
            logLikelihood = 0.0
            fn_out = tf.tensordot(matA, tQ, axes=1) 
            fp_out = tf.tensordot(matB, 1.0 - tQ, axes=1)
            logLikelihood += np.sum(fn_out + fp_out)
            self.mIndicatorQ = tQ.numpy()
            print("MFA: num. iterations = ", nIteration, logLikelihood)
            
            if np.isclose(logLikelihood, prev, rtol=0.001):
                prev = logLikelihood
                gamma *= 0.1
                nIteration = 0
            else:
                prev = logLikelihood
        return logLikelihood

    def EStep(self, mix_p, mObservationG, Nproteins, Nk, psi):
        gamma = 1000.0
        for i in range(Nproteins):
            fn_out = np.tensordot(mObservationG.mTrials[i] - mObservationG.mObserved[i], self.mIndicatorQ, axes=1) 
            fp_out = np.tensordot(psi*mObservationG.mObserved[i], 1.0 - self.mIndicatorQ, axes=1)

            mLogLikelihood = fn_out + fp_out + np.log(mix_p)
            self.mIndicatorQ[i,:] = scipy.special.softmax(-gamma*mLogLikelihood)
            
    def MStep(self, Nk, alpha):
        Z = np.sum(self.mIndicatorQ, axis=0)
        mix_p = np.zeros(Nk, dtype=float)
        for k in range(Nk):
            mix_p[k] = (Z[k] + alpha)/(np.sum(Z) + alpha*Nk)
        return mix_p

    def estimate(self, mObservationG, Nproteins, Nk, psi):
        
        print('psi = %.8f' % psi)

        mix_p = (1.0/float(Nk))*np.ones(Nk, dtype=float)
        alpha1 = 1e-8
        for i in range(Nproteins):
            self.mIndicatorQ[i] = np.random.uniform(0.0, 1.0, size=Nk)
            self.mIndicatorQ[i] = (self.mIndicatorQ[i] + alpha1)/(np.sum(self.mIndicatorQ[i]) + alpha1*Nproteins)

        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 0:
            return self.tf_annealing(mix_p, mObservationG, Nproteins, Nk, psi)
        else:
            return self.annealing(mix_p, mObservationG, Nproteins, Nk, psi)

    ##
    ## Adapt from https://github.com/zib-cmd/cmdtools/blob/dev/src/cmdtools/analysis/optimization.py
    ##
    def inner_simplex_algorithm(self, X):
        """
        Return the transformation A mapping those rows of X
        which span the largest simplex onto the unit simplex.
        """
        ind = self.indexsearch(X)
        return np.linalg.inv(X[ind, :])

    def indexsearch(self, X):
        """ Return the indices to the rows spanning the largest simplex """

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
                X /= rownorm[ind[j]]
                v  = X[ind[j], :]
                X -= np.outer(X.dot(v), v)

        return ind

    def find_lin_dependent(self):
        N = np.size(self.mIndicatorQ, axis=0)
        k = np.size(self.mIndicatorQ, axis=1)
        self.indicatorVec = np.zeros(N, dtype=int)
        ind = self.indexsearch(self.mIndicatorQ)
        for id in ind:
            for i in range(0,N):
                inner_product = np.inner(
                    self.mIndicatorQ[i],
                    self.mIndicatorQ[id])

                norm_i = np.linalg.norm(self.mIndicatorQ[i])
                norm_j = np.linalg.norm(self.mIndicatorQ[id])

                distance = np.abs(inner_product - norm_i*norm_j)
                if distance < 1E-5:
                    self.indicatorVec[i] = id
            self.indicatorVec[id] = id

    def find_argmax(self):
        N = np.size(self.mIndicatorQ, axis=0)
        k = np.size(self.mIndicatorQ, axis=1)
        self.indicatorVec = np.argmax(self.mIndicatorQ, axis=1)

    def compute_psi(self, fp, fn):
        return (np.log(1.0 - fn) - np.log(fp))/(np.log(1.0 - fp) - np.log(fn))

    def computeErrorRate(self, psi, mObservationG, Nproteins):
        
        # self.find_lin_dependent()
        self.find_argmax()

        rnk = np.linalg.matrix_rank(self.mIndicatorQ)
        print("Indicator matrix had rank = " + str(rnk))
        nClusters = len(np.unique(self.indicatorVec))
        print("Number of clusters used: " + str(nClusters))

        num_trials = 0
        countFn = 0
        countFp = 0
        trialFn = 0
        trialFp = 0
        for i in range(Nproteins):
            for j in mObservationG.lstAdjacency[i]:
                t = mObservationG.mTrials[i][j]
                s = mObservationG.mObserved[i][j]
                assert(s <= t)
                if (self.indicatorVec[i] == self.indicatorVec[j]):
                    countFn += (t - s)
                    trialFn += t
                else:
                    countFp += s
                    trialFp += t
                num_trials += t

        fn = float(countFn)/float(trialFn)
        fp = float(countFp)/float(trialFp)       
        total_loss = 0.0
        for i in range(Nproteins):
            for j in mObservationG.lstAdjacency[i]:
                t = mObservationG.mTrials[i][j]
                s = mObservationG.mObserved[i][j]
                total_loss += -s*(np.log(1.0 - fn)) - (t-s)*np.log(1.0 - fp)
        return (fn, fp, total_loss)

    def estimator_summary(self, regr, y_actual, y_pred):
        # The coefficients
        print("Coefficients: \n", regr.coef_)
        # The mean squared error
        print("Mean squared error: %.2f" % mean_squared_error(y_actual, y_pred))
        # The coefficient of determination: 1 is perfect prediction
        print("Coefficient of determination: %.2f" % r2_score(y_actual, y_pred))
        
    def computeResidues(self, mObservationG, Nproteins, Nk):
    
        (fn, fp, errs, _) = self.computeErrorRate(mObservationG, Nproteins)

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

        #result = scipy.stats.cramervonmises_2samp(residues, expectedErrors)
        #print(result)

        self.expectedErrors = expectedErrors
        self.mResidues = residues

        """ Not using statmodel package 
            # Ordinary Least Square
            est = sm.OLS(residues,  X)
            est2 = est.fit()
            print(est2.summary())

            glm_poisson = sm.GLM(residues, X, family=sm.families.Poisson(sm.families.links.log()))
            glm_results = glm_poisson.fit()
            print(glm_results.summary())
        """ 
        X = np.reshape(expectedErrors, (-1,1))
               
        # Create linear regression object
        regr = linear_model.LinearRegression()
        # Train the model using the training sets
        regr.fit(X, residues)
        # Make predictions using the testing set
        y_pred = regr.predict(X)
        print("Linear Regression evaluation:")
        self.estimator_summary(regr, residues, y_pred)
        
        return (fn, fp)

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
