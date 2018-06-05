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

class FisherClassificator(Classificator):
    Sw = np.zeros((2, 2))
    m1 = np.zeros(2)
    m2 = np.zeros(2)
    w = None

    def __init__(self):
        Classificator.__init__(self)

    def findW(self, class1, class2): 
        S1 = self.calculateClass1(class1)
        S2 = self.calculateClass2(class2)
        self.Sw = S1 + S2
        print(self.Sw)
        self.w = np.dot(inv(self.Sw), self.m1 - self.m2)


    def calculateClass1(self, class1):
        self.m1, S1 = self.calculateClass(class1)
        return S1

    def calculateClass2(self, class2):   
        self.m2, S2 = self.calculateClass(class2)
        return S2

    def calculateClass(self, cl):
        ni = len(cl)
        cl_arr = np.array(cl)
        mi = np.sum(cl_arr, axis = 0) / ni
        V = [x - mi for x in cl_arr]
        Si = sum([np.outer(v, v) for v in V])
        print(mi)
        print(Si)
        return mi, Si

    def classificate(self, x):
        return 0 if np.dot(self.w, x) > 0.5 * np.dot(self.w, self.m1 + self.m2) else 1
        """
        y1 = np.dot(self.w, x - self.m1)
        y2 = -np.dot(self.w, x - self.m2)
        print(y1)
        print(y2)
        return 0 if y1 > y2 else 1
        """

class MCFisherClassficator(Classificator):
    W = []
    eigVal = []
    means = []

    def __init__(self):
        Classificator.__init__(self)

    def findW(self, classes):
        data = [self.calculateClass(cl) for cl in classes]
        self.means  = [val[0] for val in data]
        m = sum([val[0] * val[1] for val in data]) / sum([val[1] for val in data])
        Sw = sum([val[2] for val in data])
        Sb = sum([np.outer(val[2] - m, val[2] - m) for val in data])
        self.W, self.eigVal = np.linalg.eig(np.dot(inv(Sw), Sb)) 
        print(self.W)
        print(self.eigVal)

    def calculateClass(self, cl):
        ni = len(cl)
        cl_arr = np.array(cl)
        mi = np.sum(cl_arr, axis = 0) / ni
        V = [x - mi for x in cl_arr]
        Si = sum([np.outer(v, v) for v in V])
        print(mi)
        print(Si)
        return [mi, ni, Si]

    def classificate(self, x):
        y = np.dot(self.W, x)
        Y = np.dot(self.W, self.means)
        #?