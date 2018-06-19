#!/usr/bin/python
import numpy as np
from numpy.linalg import inv
import math

class Fisher():
    Sw = np.zeros((2, 2))
    m1 = np.zeros(2)
    m2 = np.zeros(2)
    w = None

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

    def reduceDimension(self, x):
        return np.dot(self.w, x)

    def classificate(self, x):
        return 0 if self.reduceDimension(x) > 0.5 * np.dot(self.w, self.m1 + self.m2) else 1
        """
        y1 = np.dot(self.w, x - self.m1)
        y2 = -np.dot(self.w, x - self.m2)
        print(y1)
        print(y2)
        return 0 if y1 > y2 else 1
        """

class MCFisher():
    W = []
    eigVal = []
    means = []

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
        cl = np.array(cl)
        mi = np.sum(cl, axis = 0) / ni
        V = [x - mi for x in cl]
        Si = sum([np.outer(v, v) for v in V])
        print(mi)
        print(Si)
        return [mi, ni, Si]

    def reduceDimension(self, x):
        y = np.dot(self.W, x)
        return y
    