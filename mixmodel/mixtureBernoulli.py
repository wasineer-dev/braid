import numpy as np
import scipy.special

from meanfield.simulateLikelihood import MAX_ITERATION

class MixtureBernoulli:

    def __init__(self, mObservationG, psi):
        self.matA = mObservationG.mTrials - mObservationG.mObserved
        self.matB = psi*mObservationG.mObserved
        
    def EStep(self, Xs, p, mix_p):
        Ns, nD = Xs.shape
        mLogLikelihood  = (Xs @ np.log(p)) + ((1.0 - Xs) @ np.log(1.0-p))
        Z = np.tile(np.log(mix_p), (Ns, 1)) + mLogLikelihood
        self.eta = scipy.special.softmax(Z, axis=1)

    def EStepMFA(self, Xs, p, mix_p, gamma):
        Ns, nD = Xs.shape
        nD, nK = p.shape

        mLogLikelihood  = (Xs @ np.log(p)) + ((1.0 - Xs) @ np.log(1.0-p))
        Z = np.tile(np.log(mix_p), (Ns, 1)) + mLogLikelihood
        # Dependency
        for i in range(Ns):
            fn_out = np.tensordot(self.matA[i], self.eta, axes=1)
            fp_out = np.tensordot(self.matB[i], 1.0 - self.eta, axes=1)
            Z[i] += (fn_out + fp_out)
            self.eta[i] = scipy.special.softmax(-gamma*Z[i])
        
        """
        for k in range(nK):
            Z[k] = np.log(mix_p[k]) + np.dot(Xs[i], np.log(p[:,k])) + np.dot(1.0 - Xs[i], np.log(1.0 - p[:,k]))
        """
    #
    def MStep(self, Xs, alpha1, alpha2):
        Ns, nD = Xs.shape
        ns, Nk = self.eta.shape

        p = np.transpose(Xs) @ self.eta
        Z = np.sum(self.eta, axis=0)
        for k in range(Nk):
            """
            for k in range(Nk):
                p[d][k] = np.dot(self.eta[:,k], Xs[:,d]) + alpha1
            """
            p[:,k] = (p[:,k] + alpha1)/(Z[k] + alpha1*nD)

        mix_p = np.zeros(Nk, dtype=float)
        for k in range(Nk):
            mix_p[k] = (Z[k] + alpha2)/(np.sum(Z) + alpha2*Nk)

        return (p, mix_p) 

    def estimate(self, Xs, Nk, alpha1, alpha2):
        Ns, nD = Xs.shape
        p = np.zeros(shape=(nD, Nk), dtype=float)
        mix_p = (1.0/float(Nk))*np.ones(Nk, dtype=float)
        for i in range(nD):
            p[i] = np.random.uniform(0.0, 1.0, size=Nk)
            p[i] = (p[i] + alpha1)/(np.sum(p[i]) + alpha1*nD)

        # Run EM without MFA to initialize
        nIteration = 20
        for n in range(nIteration):
            self.EStep(Xs, p, mix_p)
            p, mix_p = self.MStep(Xs, alpha1, alpha2)

        nIteration = 0
        gamma = 1000.0
        while(nIteration < MAX_ITERATION and gamma > 1.0):
            self.EStepMFA(Xs, p, mix_p, gamma)
            p, mix_p = self.MStep(Xs, alpha1, alpha2)
            nIteration += 1
            gamma = gamma - 100

        return (p, mix_p)

    def predict(self, Xs, p, mix_p):
        Ns, nD = Xs.shape
        nD, Nk = p.shape
        
        self.EStep(Xs, p, mix_p)

        y_pred = np.argmax(self.eta, axis=1)
        return y_pred