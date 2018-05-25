#!/usr/bin/python
import numpy as np
from numpy.linalg import inv
import math

class Classificator:
    W = None

    def findW(self, classes):
        T = []
        X = []

        for i in xrange(len(classes)):
            cl = classes[i]
            t = [0] * len(classes)
            t[i] = 1
            [X.append([1, x1, x2]) for x1, x2 in zip(cl[0], cl[1])]
            T.extend([t] * len(cl[0]))
        
        X_matrix = np.array(X)
        Xt = X_matrix.T
        try:
            XtXInv = inv(np.dot(Xt, X))
            X_star = np.dot(XtXInv, Xt)
            self.W = np.dot(X_star, T)
            print(self.W)
        except np.linalg.LinAlgError as e:
            print(e)

    def classificate(self, x1, x2):
        Y = np.dot(self.W, [1, x1, x2])
        return np.argmax(Y)