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

    def classificate(self, x1, x2):
        Y = np.dot(self.W.T, [1, x1, x2])
        print(Y)
        return np.argmax(Y)

class FisherClassificator(Classificator):
    Sw = np.zeros((2, 2))
    m1 = np.zeros(2)
    m2 = np.zeros(2)

    def __init__(self):
        Classificator.__init__(self)

    def findW(self, class1, class2): 
        self.calculateMean1(class1)
        self.calculateMean2(class2)

    def calculateMean1(self, class1):
        self.m1 = self.calculateMean(class1)
        
    def calculateMean2(self, class2):   
        self.m2 = self.calculateMean(class2)

    def calculateMean(self, cl):
        n = len(cl)
        cl_arr = np.array(cl)
        mi = np.sum(cl_arr, axis = 0) / n
        V = [x - mi for x in cl_arr]
        Si = sum([np.outer(v, v) for v in V])
        print(mi)
        print(Si)
        self.Sw = self.Sw + Si
        print(self.Sw)
        return mi