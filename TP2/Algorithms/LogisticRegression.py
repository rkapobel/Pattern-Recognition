#!/usr/bin/python
import numpy as np
from numpy.linalg import inv
import math

LINEAR = lambda x: [1, x[0], x[1]]
CIRCULAR = lambda x: [1, x[0]**2, x[1]**2]
ELLIPTIC = lambda x: [1, x[0], x[1], x[0]**2, x[0]*x[1], x[1]**2]

class LogisticRegression:

    def __init__(self, phi, maxIter = 200):
        self.costs = []
        self.epochs = []
        self.phi = phi
        self.maxIter = maxIter
    
    def sigmoid(self, gamma):
        sigmoids = []
        #TODO: Figure out if it is possible to use np.exp
        for a in gamma:
            if a < 0:
                sigmoids.append(1 - 1 / (1 + np.exp(a)))
            else:
                sigmoids.append(1 / (1 + np.exp(-a)))
        return np.array(sigmoids)

    def calculateCostFunction(self, w, phi_X, labels, numIter):
        m = phi_X.shape[0]
        h = self.sigmoid(np.dot(phi_X, w))
        labels = np.array(labels)
        return self.costFunction(h, labels, m)
        
    def costFunction(self, h, labels, m):
        return (np.dot(-1 * labels, np.log(h)) - np.dot(1 - labels, np.log(1 - h))) / float(m)

    def getEpochs(self):
        return list(range(self.maxIter))

class NRLogisticRegression(LogisticRegression):
    
    def __init__(self):
        self.w = None

    def findW(self, classes):
        phi_X = []
        t = []

        for k in xrange(len(classes)):
            ck = classes[k]
            [phi_X.append(self.phi(x)) for x in ck] #NxM (phi: NxD -> NxM)
            t.extend([1 - k] * len(ck))

        phi_X = np.array(phi_X)  
        t = np.array(t)
        n, m = phi_X.shape
        w_old = np.zeros((m,)) # Did not like to start with random
        self.w = np.array([float('inf')] * m)
        phi_X_t = phi_X.T #MxN
        numIter = 0 
        allCosts = []
        while numIter <= self.maxIter:
            y = self.sigmoid(np.dot(phi_X, w_old)) #Nx1
            R = np.diag([s * (1 - s) for s in y]) #NxN
            L = np.dot(phi_X_t, R) #MxN
            try:
                M = inv(np.dot(L, phi_X)) #MxNxNxM
                z = np.dot(phi_X, w_old) - np.dot(inv(R), y - t)
                w_old = np.dot(M, np.dot(L, z))
                cost = self.calculateCostFunction(w_old, phi_X, t, numIter)
                allCosts.append(cost)
            except np.linalg.LinAlgError as e:
                print(e)
                break
            numIter += 1
        self.w = w_old
        self.costs.append(allCosts)

    def classificate(self, x):
        return 0 if self.sigmoid([np.dot(self.w, self.phi(x))]) >= 0.05 else 1

class MCLogisticRegression(LogisticRegression):

    def __init__(self, phi, maxIter = 200, regularize = False, alpha = 0.005):
        LogisticRegression.__init__(self, phi, maxIter)
        self.W = None
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
        allCosts = []
        for k in xrange(self.K if self.K > 2 else 1):
            numIter = 0
            while numIter <= self.maxIter:    
                wk = W_old[k] if self.K > 2 else W_old
                diff =  self.sigmoid(np.dot(phi_X, wk)) - T.T[k]
                j = 0
                gradient = np.zeros((m,))
                for x in phi_X:
                    gradient += x * diff[j]
                    j += 1
                
                if self.regularize:
                    regTerm = wk
                    regTerm[0] = 0
                    wk -= (self.alpha / float(n)) * (gradient + regTerm)
                else:
                    wk -= (self.alpha / float(n)) * gradient

                if self.K > 2:
                    W_old[k] = wk
                else: 
                    W_old = wk

                cost = self.calculateCostFunction(wk, phi_X, T[:, k if self.K > 2 else 0], numIter)
                allCosts.append(cost)

                numIter += 1
        self.W = W_old

        for i in range(0, len(allCosts), self.maxIter + 1):
            self.costs.append(allCosts[i:i+self.maxIter])

        print('W: {0}'.format(self.W))
                
    def classificate(self, x):
        if self.K == 2:
            return 0 if self.sigmoid([np.dot(self.W, self.phi(x))]) >= 0.5 else 1
        return np.argmax(self.sigmoid(np.dot(self.W, self.phi(x))))