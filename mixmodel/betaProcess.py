from re import M
import numpy as np
import scipy.special

from meanfield.simulateLikelihood import MAX_ITERATION
from mixmodel.mixtureBernoulli import MixtureBernoulli

class BetaProcess:

    def __init__(self, mObservationG, Xs, Nk, psi=None):
        
        Ns, Nd = Xs.shape

        self.alpha = 1.0
        self.beta = 1.0
        # Initialize covariance matrix
        Xs += np.random.normal(size=(Ns*Nd)).reshape(Ns,Nd)
        self.cov = np.cov(Xs.T)
        self.inverseCov = np.linalg.inv(self.cov)
        # self.inverseCov = np.diag(1.0/np.diag(self.cov))  # TODO: Use Variance of Binomial distribution?
        # Generate factor loading matrix
        self.theta = np.random.multivariate_normal(np.zeros(Nd, dtype=float), self.cov, size=Nk)
        
        self.mixBernoulli = MixtureBernoulli(mObservationG)

    def precompute(self):
        Ns, Nk = self.eta.shape
        Nk, nD = self.theta.shape

        vecB = self.eta @ self.theta
        tensorB = np.tile(vecB, (Nk, 1)).reshape(Nk,Ns,nD)
        for k in range(Nk): 
            for i in range(Ns):
                tensorB[k][i] -= self.eta[i][k]*self.theta[k]    
        return tensorB

    def EStep(self, Xs, alpha, beta):
        Ns, nD = Xs.shape
        Nk, nD = self.theta.shape

        tensorB = self.precompute()
        vecA = np.zeros(Nk, dtype=float)
        for k in range(Nk):
            thisCov = (self.Zk[k]*np.diag(np.ones(nD)) + self.inverseCov)
            vecSigma = np.linalg.inv(thisCov)
            vecA[k] = np.dot(self.theta[k], self.theta[k]) + np.trace(vecSigma)

        for i in range(Ns):
            mLogLikelihood = scipy.special.digamma((alpha/float(Nk)) + self.Zk) - scipy.special.digamma((alpha + beta*(Nk-1))/float(Nk)  + Ns)
            for k in range(Nk):
                vecC = Xs[i] - tensorB[k][i]
                mLogLikelihood[k] += -0.5*(vecA[k] - 2.0*(np.dot(self.theta[k],vecC)))
            self.eta[i] = mLogLikelihood
        
        # Normalize with softmax
        self.eta = scipy.special.softmax(self.eta, axis=1)
        del tensorB

    def MStep(self, Xs):
        Ns, Nk = Xs.shape
        Nk, nD = self.theta.shape

        tensorB = self.precompute()
        
        self.Zk = np.sum(self.eta, axis=0)

        for k in range(Nk):
            vecMu = np.sum(self.Zk[k]*(Xs - tensorB[k]), axis=0)
            thisCov = (self.Zk[k]*np.diag(np.ones(nD)) + self.inverseCov)
            thisCov = np.linalg.inv(thisCov)
            self.theta[k] = np.dot(vecMu, thisCov) 
        del tensorB

    def estimate(self, Xs, Nk):
        
        p, mix_p = self.mixBernoulli.estimate(Xs, Nk, 1e-8, 1e-8)
        self.eta = self.mixBernoulli.eta
        self.Zk = np.sum(self.eta, axis=0)

        nIteration = 5
        for n in range(nIteration):
            self.EStep(Xs, self.alpha, self.beta)
            self.MStep(Xs)
        return mix_p

    def predict(self, Xs, mix_p):
        self.EStep(Xs, 1.0, 1.0)
        Ns, Nk = self.eta.shape
        y_pred = np.zeros((Ns, Nk), dtype=int)
        for i in range(Ns):
            p = np.random.rand()
            y_pred[i] = self.eta[i] > p 
        return y_pred