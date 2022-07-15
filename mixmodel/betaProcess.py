from re import M
import numpy as np
import scipy.special

from meanfield.simulateLikelihood import MAX_ITERATION
from mixmodel.mixtureBernoulli import MixtureBernoulli

class BetaProcess:

    def __init__(self, mObservationG, Xs, Nk, psi=None):
        Nd, Ns = Xs.shape

        self.alpha = 1.0
        self.beta = 1.0
        # Initialize covariance matrix
        self.cov = np.cov(Xs)
        self.inverseCov = np.linalg.inv(self.cov)
        # self.inverseCov = np.diag(1.0/np.diag(self.cov))  # TODO: Use Variance of Binomial distribution?
        # Generate factor loading matrix
        self.theta = np.random.multivariate_normal(np.zeros(Nd, dtype=float), self.cov, size=Nk)
        
    def precompute(self, Xs):
        Ns, Nk = self.eta.shape
        Nk, nD = self.theta.shape

        """
        tensorB = np.tile(Xs, (Nk, 1)).reshape(Nk,nD,Ns)
        for k in range(Nk):
            for i in range(nD):
                tensorB[k][i] -= self.theta[k][i] * self.eta[:,k]
        """
        vecResult = np.zeros((Ns, Nk), dtype=float)
        for j in range(Ns):
            for k in range(Nk):
                thetaK = np.tile(self.theta[k], (Ns,1)).T
                vecC = Xs[:,j] - np.dot(thetaK, self.eta[:,k]) 
                #for i in range(nD):
                #    vecC[i] -= self.theta[k][i] * self.eta[:,k]        
                vecResult[j][k] = np.dot(self.theta[k], vecC)      
        return vecResult

    def EStep(self, Xs, alpha, beta):
        Nd, Ns = Xs.shape
        Nk, nD = self.theta.shape

        vecA = np.zeros(Nk, dtype=float)
        for k in range(Nk):
            thisCov = (self.Zk[k]*np.diag(np.ones(nD)) + self.inverseCov)
            vecSigma = np.linalg.inv(thisCov)
            vecA[k] = np.dot(self.theta[k], self.theta[k]) + np.trace(vecSigma)

        vecC = self.precompute(Xs)
        for i in range(Ns):
            mLogLikelihood = scipy.special.digamma((alpha/float(Nk)) + self.Zk) - scipy.special.digamma((alpha + beta*(Nk-1))/float(Nk)  + Ns)
            for k in range(Nk):
                mLogLikelihood[k] += -0.5*(vecA[k] - 2.0*vecC[i,k])
            self.eta[i] = mLogLikelihood
        
        tmp = np.zeros(shape=(Ns, Nk), dtype=float)
        for i in range(Ns):
            tmp[i] = scipy.special.digamma((beta*(Nk-1)/float(Nk)) + Ns - self.Zk) - scipy.special.digamma((alpha + beta*(Nk-1))/float(Nk)  + Ns)
        
        self.eta = scipy.special.expit(-self.eta)
        self.eta = self.eta/(self.eta + scipy.special.expit(-tmp))

    def MStep(self, Xs):
        Nd, Ns = Xs.shape
        Nk, Nd = self.theta.shape

        # tensorB = self.precompute(Xs)
        
        self.Zk = np.sum(self.eta, axis=0)

        for k in range(Nk):
            thetaK = np.tile(self.theta[k], (Ns,1)).T
            vecC = Xs.T
            for j in range(Ns):
                vecC[j] -= np.dot(thetaK, self.eta[:,k]) 
            #for j in range(Nd):
            #    vecC[j] -= self.theta[k][j] * self.eta[:,k]
            vecMu = np.dot(self.eta[:,k], vecC)
            thisCov = (self.Zk[k]*np.diag(np.ones(Nd)) + self.inverseCov)
            thisCov = np.linalg.inv(thisCov)
            self.theta[k] = np.dot(vecMu, thisCov) 

    def estimate(self, Xs, Nk):
        Nd, Ns = Xs.shape
        mix_p = np.random.beta(1.0/float(Nk), 1.0*(Nk-1)/float(Nk), size=Nk)
        self.eta = np.zeros(shape=(Ns, Nk), dtype=float)
        for i in range(Ns):
            self.eta[i] = np.random.binomial(1, mix_p)
        self.Zk = np.sum(self.eta, axis=0)
        
        nIteration = 2
        for n in range(nIteration):
            self.EStep(Xs, self.alpha, self.beta)
            self.MStep(Xs)

    def predict(self, Xs):
        self.EStep(Xs, self.alpha, self.beta)
        Ns, Nk = self.eta.shape
        y_pred = np.zeros((Ns, Nk), dtype=int)
        for i in range(Ns):
            y_pred[i] = self.eta[i] > np.random.rand(Nk) 
        return y_pred