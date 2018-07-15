#!/usr/bin/python
import numpy as np
from numpy.linalg import inv
import math

LINEAR = lambda x: [1, x[0], x[1]]
CIRCULAR = lambda x: [1, x[0]**2, x[1]**2]
ELLIPTIC = lambda x: [1, x[0], x[1], x[0]**2, x[0]*x[1], x[1]**2]

class LogisticRegression:
    phi = None
    maxIter = 200

    def __init__(self, phi, maxIter = 200):
        self.phi = phi
        self.maxIter = maxIter
    
    def sigmoid(self, gamma):
        if gamma < 0:
            return 1 - 1 / (1 + math.exp(gamma))
        return 1 / (1 + math.exp(-gamma))

class NRLogisticRegression(LogisticRegression):
    w = None

    def findW(self, classes):
        phi_X = []
        t = []

        for i in xrange(len(classes)):
            ci = classes[i]
            [phi_X.append(self.phi(x)) for x in ci] #NxM (phi: NxD -> NxM)
            t.extend([1 - i] * len(ci))

        phi_X = np.array(phi_X)  
        t = np.array(t)
        n, m = phi_X.shape
        w_old = np.zeros((m,)) # Did not like to start with random
        self.w = np.array([float('inf')] * m)
        phi_X_t = phi_X.T #MxN
        numIter = 0
        while numIter <= self.maxIter:
            y = np.array([self.sigmoid(np.dot(w_old, p)) for p in phi_X]) #Nx1
            R = np.diag([s * (1 - s) for s in y]) #NxN
            L = np.dot(phi_X_t, R) #MxN
            try:
                M = inv(np.dot(L, phi_X)) #MxNxNxM
                z = np.dot(phi_X, w_old) - np.dot(inv(R), y - t)
                self.w = w_old
                w_old = np.dot(M, np.dot(L, z))
            except np.linalg.LinAlgError as e:
                print(e)
                break
            numIter += 1
        self.w = w_old

    def classificate(self, x):
        return 0 if self.sigmoid(np.dot(self.w, self.phi(x))) >= 0.5 else 1

class MCLogisticRegression(LogisticRegression):
    W = None
    regularize = False
    alpha = 1

    def __init__(self, phi, maxIter = 200, regularize = False, alpha = 1):
        LogisticRegression.__init__(self, phi, maxIter)
        self.regularize = regularize
        self.alpha = alpha

    def findW(self, classes):
        phi_X = []
        T = []

        self.K = len(classes)
        for i in xrange(self.K):
            ci = classes[i]
            [phi_X.append(self.phi(x)) for x in ci] #NxM (phi: NxD -> NxM)
            t = [0] * self.K
            t[i] = 1
            T.extend([t] * len(ci))

        phi_X = np.array(phi_X) #NxM
        T = np.array(T) #NxK
        n, m = phi_X.shape
        W_old = np.zeros((self.K, m) if self.K > 2 else (m,)) #KxM
        
        for i in xrange(self.K if self.K > 2 else 1):
            numIter = 0
            while numIter <= self.maxIter:    
                wi = W_old[i] if self.K > 2 else W_old
                gammas = np.dot(phi_X, wi)
                sigmoids = [self.sigmoid(gamma) for gamma in gammas]
                diff =  sigmoids - T.T[i]
                j = 0
                gradient = np.zeros((m,))
                for x in phi_X:
                    gradient += x * diff[j]
                    j += 1
                
                if self.regularize:
                    regTerm = wi
                    regTerm[0] = 0
                    wi -= (self.alpha / float(n)) * (gradient + regTerm)
                else:
                    wi -= (self.alpha / float(n)) * gradient

                if self.K > 2:
                    W_old[i] = wi
                else: 
                    W_old = wi

                numIter += 1
        self.W = W_old
        print('W: {0}'.format(self.W))
                
    def classificate(self, x):
        if self.K == 2:
            gamma = np.dot(self.W, self.phi(x))
            return 0 if gamma >= 0.5 else 1

        gammas = np.dot(self.W, self.phi(x))
        sigmoids = []
        for gamma in gammas:
            sigmoids.append(self.sigmoid(gamma))
        return np.argmax(sigmoids)