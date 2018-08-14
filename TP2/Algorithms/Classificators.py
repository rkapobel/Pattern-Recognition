#!/usr/bin/python
import numpy as np
from numpy.linalg import inv
import math

class Classificator:

    def __init__(self):
       self.W = None

class LinearClassificator(Classificator):

    def __init__(self):
        Classificator.__init__(self)

    def findW(self, classes):
        T = []
        X = []

        for i in xrange(len(classes)):
            ci = classes[i]
            t = [0] * len(classes)
            t[i] = 1
            [X.append([1, x[0], x[1]]) for x in ci]
            T.extend([t] * len(ci))
        
        X_matrix = np.array(X)
        Xt = X_matrix.T
        try:
            XtXInv = inv(np.dot(Xt, X))
            X_star = np.dot(XtXInv, Xt)
            self.W = np.dot(X_star, T)
            print('W: {0}'.format(self.W))
        except np.linalg.LinAlgError as e:
            print(e)

    def classificate(self, x):
        Y = np.dot(self.W.T, [1, x[0], x[1]])
        print('Y: {0}'.format(Y))
        return np.argmax(Y)