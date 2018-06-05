#!/usr/bin/python
import numpy as np
from numpy.linalg import inv
import math

class LogisticRegression:
    phi = None
    w = None

    def __init__(self, phi):
        self.phi = phi        

    def findW(self, data, t):
        phi_data = self.phi(np.array(data))         
        w = np.random.normal(0, 1, len(phi_data[0]))
        for _ in xrange(100): # threshold is number of iterations and distance between w_n and w_o
            y = [1 / (1 + np.exp(np.dot(w, p))) for p in phi_data]
            R = np.diag([s * (1 - s) for s in y])
            phi_data_t = phi_data.T
            L = np.dot(phi_data_t, R)
            M = inv(np.dot(L, phi_data))
            z = np.dot(phi_data, w) - np.dot(inv(R), y - t)
            w = np.dot(M, np.dot(L, z))

    def y(self, x):
        prob = 1 / (1 + np.exp(np.dot(self.w, self.phi(x))))
        return 1 if prob >= 0.5 else 0