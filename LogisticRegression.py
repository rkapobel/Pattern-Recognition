#!/usr/bin/python
import numpy as np
from numpy.linalg import inv
import math

threshold = .5

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
        w_old = np.zeros(len(phi_X[0]))
        self.w = np.array([float('inf')] * len(phi_X[0]))
        phi_X_t = phi_X.T #2xN
        while np.linalg.norm(self.w - w_old, ord = 2) / (1.0 * len(phi_X))  > threshold:
            y = np.array([1 / (1 + np.exp(float(-np.dot(w_old, p)))) for p in phi_X]) #Nx1
            R = np.diag([s * (1 - s) for s in y]) #NxN
            L = np.dot(phi_X_t, R) #2xN
            try:
                M = inv(np.dot(L, phi_X))
                z = np.dot(phi_X, w_old) - np.dot(inv(R), y - t)
                self.w = w_old
                w_old = np.dot(M, np.dot(L, z))
            except np.linalg.LinAlgError as e:
                print(e)
                break
        self.w = w_old

    def classificate(self, x):
        x_aux = [1]
        x_aux.extend(x)
        prob = 1 / (1 + np.exp(float(np.dot(self.w, self.phi(x_aux)))))
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
        W_old = np.zeros([T.shape[1], phi_X.shape[1]]) #KxM
        self.W = float('inf') + W_old
        ita = 0.5
        while np.linalg.norm(self.W[0] - W_old[0]) / (1.0 * len(phi_X)) > threshold:
            for i in xrange(W_old.shape[0]):
                m = np.dot(phi_X, W_old[i])
                v = [1 / (1 + np.exp(float(-wp))) for wp in m] - T.T[i]
                j = 0
                grad = np.zeros((phi_X.shape[1],))
                for row in phi_X:
                    grad += row * v[j]
                    j += 1
                self.W[i] = W_old[i]
                W_old[i] -= ita * grad
        self.W = W_old
                
    def classificate(self, x):
        x_aux = [1]
        x_aux.extend(x)
        return np.argmax(np.dot(self.W, x_aux))