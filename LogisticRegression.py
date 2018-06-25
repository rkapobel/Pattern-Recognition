#!/usr/bin/python
import numpy as np
from numpy.linalg import inv
import math

class LogisticRegression:
    phi = None
    w = None

    def __init__(self, phi):
        self.phi = phi        

    def findW(self, classes):
        X = []
        t = []

        for i in xrange(len(classes)):
            ci = classes[i]
            [X.append([1, x[0], x[1]]) for x in ci]
            t.extend([1 - i] * len(ci))

        phi_X = self.phi(np.array(X))  
        t = np.array(t)
        self.w = np.zeros(len(phi_X[0]))
        phi_X_t = phi_X.T #2xN
        for itNum in xrange(100): # threshold is number of iterations and distance between w_n and w_o
            y = np.array([1 / (1 + np.exp(-np.dot(self.w, p))) for p in phi_X]) #Nx1
            R = np.diag([s * (1 - s) for s in y]) #NxN
            L = np.dot(phi_X_t, R) #2xN
            print(self.w)
            try:
                M = inv(np.dot(L, phi_X))
                z = np.dot(phi_X, self.w) - np.dot(inv(R), y - t)
                self.w = np.dot(M, np.dot(L, z))
            except np.linalg.LinAlgError as e:
                print(e)
                break

    def classificate(self, x):
        x_aux = [1]
        x_aux.extend(x)
        prob = 1 / (1 + np.exp(np.dot(self.w, self.phi(x_aux))))
        return 1 if prob >= 0.5 else 0

class MCLogisticRegression:
    phi = None
    W = None

    def __init__(self, phi):
        self.phi = phi 

    def findW(self, classes):
        X = []
        T = []

        for i in xrange(len(classes)):
            ci = classes[i]
            [X.append([1, x[0], x[1]]) for x in ci]
            t = [0] * len(classes)
            t[i] = 1
            T.extend([t] * len(ci))

        X = np.array(X) #NxD
        T = np.array(T) #NxK
        phi_X = self.phi(X) #NxM (phi: NxD -> NxM)
        self.W = np.zeros([T.shape[1], phi_X.shape[1]]) #KxM
        ita = 0.5
        for itNum in xrange(100):
            for i in xrange(self.W.shape[0]):
                m = np.dot(phi_X, self.W[i])
                v = [1 / (1 + np.exp(-wp)) for wp in m] - T.T[i]
                j = 0
                grad = np.zeros((phi_X.shape[1],))
                for row in phi_X:
                    grad += row * v[j]
                    j += 1
                self.W[i] -= ita * grad
                
    def classificate(self, x):
        x_aux = [1]
        x_aux.extend(x)
        return np.argmax(np.dot(self.W, x_aux))