import numpy as np
import scipy.special

from meanfield.simulateLikelihood import MAX_ITERATION

class BetaProcess:

    def __init__(self, mObservationG, Xs, Nk):
        
        Ns, Nd = Xs.shape

        self.alpha = 1.0
        self.beta = 1.0
        # Initialize covariance matrix
        self.cov = np.cov(Xs.T)
        self.inverseCov = np.diag(1.0/np.diag(self.cov))  # TODO: Use Variance of Binomial distribution?
        # Generate factor loading matrix
        self.theta = np.random.multivariate_normal(np.zeros(Nd, dtype=float), self.cov, size=Nk)
        
    def EStep(self, Xs, mix_p, alpha, beta):
        Ns, nD = Xs.shape
        Nk, nD = self.theta.shape

        vecB = self.eta @ self.theta
        tsB = np.tile(vecB, (Nk, 1)).reshape(Nk,Ns,nD)
        for k in range(Nk): 
            for i in range(Ns):
                tsB[k][i] -= self.eta[i][k]*self.theta[k]    

        vecA = np.zeros(Nk, dtype=float)
        for k in range(Nk):
            thisCov = (self.Nk[k]*np.diag(np.ones(nD)) + self.inverseCov)
            thisCov = np.diag(1.0/np.diag(thisCov))
            vecA[k] = np.dot(self.theta[k], self.theta[k]) + np.trace(thisCov)

        # Use this to calculate P(z_ik) = 1: mLogLikelihood = scipy.special.digamma((alpha/float(Nk)) + self.Nk[k]) - scipy.special.digamma((alpha + beta*(Nk-1))/float(Nk)  + Ns)
        for i in range(Ns):
            mLogLikelihood = scipy.special.digamma((alpha/float(Nk)) + self.Nk) - scipy.special.digamma((alpha + beta*(Nk-1))/float(Nk)  + Ns)
            for k in range(Nk):
                vecC = Xs[i] - tsB[k][i]
                mLogLikelihood[k] += -0.5*(vecA[k] - 2.0*(np.dot(self.theta[k],vecC)))
            self.eta[i] = mLogLikelihood
        
        # Normalize with softmax
        self.eta = scipy.special.softmax(self.eta, axis=1)

    def MStep(self):
        self.Nk = np.sum(self.eta, axis=0)

    def estimate(self, Xs, Nk):
        Ns, nD = Xs.shape
        mix_p = (1.0/float(Nk))*np.ones(Nk, dtype=float)
        self.eta = np.zeros(shape=(Ns, Nk), dtype=int)
        for i in range(Ns):
            self.eta[i] = np.random.binomial(1, mix_p, size=Nk)
        self.Nk = np.sum(self.eta, axis=0)

        self.EStep(Xs, mix_p, self.alpha, self.beta)
        self.MStep()
        return mix_p

    def predict(self, Xs, mix_p):
        self.EStep(Xs, mix_p, 1.0, 1.0)
        Ns, Nk = self.eta.shape
        y_pred = np.zeros((Ns, Nk), dtype=int)
        for i in range(Ns):
            p = np.random.rand()
            y_pred[i] = self.eta[i] > p 
        return y_pred