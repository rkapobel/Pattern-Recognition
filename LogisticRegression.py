#!/usr/bin/python
import numpy as np
from numpy.linalg import inv
import math

class LogisticRegression:
    phi = None
    w = None

    def __init__(self, phi):
        self.phi = phi        

    def findW(self, X, t):
        phi_X = self.phi(np.array(X))         
        w = np.random.normal(0, 1, len(phi_X[0]))
        itNum = 0
        for itNum in xrange(100): # threshold is number of iterations and distance between w_n and w_o
            y = [1 / (1 + np.exp(np.dot(w, p))) for p in phi_X]
            R = np.diag([s * (1 - s) for s in y])
            phi_X_t = phi_X.T
            L = np.dot(phi_X_t, R)
            M = inv(np.dot(L, phi_X))
            z = np.dot(phi_X, w) - np.dot(inv(R), y - t)
            w = np.dot(M, np.dot(L, z))

    def y(self, x):
        prob = 1 / (1 + np.exp(np.dot(self.w, self.phi(x))))
        return 1 if prob >= 0.5 else 0

class MCLogisticRegression:
    phi = None
    W = None

    def __init__(self, phi):
        self.phi = phi 

    def findW(self, X, T):
        X = np.array(X) #NxD
        T = np.array(T) #NxK
        phi_X = self.phi(X) #NxM (phi: NxD -> NxM)
        self.W = np.zeros([T.shape[1], phi_X.shape[1]]) #KxM
        itNum = 0
        while itNum in xrange(100):
            # Can I use the softmax without using Newton Raphson?
            M = self.softmax(phi_X) - T
            M = M.T
            self.W -= np.dot(M, phi_X)
    
    def classificate(self, x):
        return np.argmax(np.dot(self.W, x))

    def softmax(self, phi_X): #NxM
        WxPhi = np.exp(np.dot(self.W, phi_X.T).T) #(KxMxMxN)t = NxK
        softmaxM = np.array(WxPhi.shape)
        sumWxPhiRows = sum(WxPhi)
        softmaxM = WxPhi / sumWxPhiRows.reshape((sumWxPhiRows.shape[0], 1))
        return softmaxM #NxK