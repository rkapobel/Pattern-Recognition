#!/usr/bin/python
import numpy as np
from numpy.linalg import inv
import math

class Classificator:
    W = None

    def __init__(self):
        pass

class LinearClassificator(Classificator):
    def __init__(self):
        Classificator.__init__(self)

    def findW(self, classes):
        T = []
        X = []

        for i in xrange(len(classes)):
            cl = classes[i]
            t = [0] * len(classes)
            t[i] = 1
            [X.append([1, x[0], x[1]]) for x in cl]
            T.extend([t] * len(cl))
        
        X_matrix = np.array(X)
        Xt = X_matrix.T
        try:
            XtXInv = inv(np.dot(Xt, X))
            X_star = np.dot(XtXInv, Xt)
            self.W = np.dot(X_star, T)
            print(self.W)
        except np.linalg.LinAlgError as e:
            print(e)

    def classificate(self, x):
        Y = np.dot(self.W.T, [1, x[0], x[1]])
        print(Y)
        return np.argmax(Y)