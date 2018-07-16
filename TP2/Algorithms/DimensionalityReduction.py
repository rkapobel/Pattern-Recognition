#!/usr/bin/python
import numpy as np
from numpy.linalg import inv
import math

class Fisher():
    Sw = np.zeros((2, 2))
    m1 = np.zeros(2)
    m2 = np.zeros(2)
    w = None

    def findW(self, data1, data2): 
        S1 = self.calculateClass1(data1)
        S2 = self.calculateClass2(data2)
        self.Sw = S1 + S2
        self.w = np.dot(inv(self.Sw), self.m1 - self.m2)
        print('w: {0}'.format(self.w))

    def calculateClass1(self, data1):
        self.m1, S1 = self.calculateClass(data1)
        return S1

    def calculateClass2(self, data2):   
        self.m2, S2 = self.calculateClass(data2)
        return S2

    def calculateClass(self, ci):
        ni = len(ci)
        ci_arr = np.array(ci)
        mi = np.sum(ci_arr, axis = 0) / ni
        V = [x - mi for x in ci_arr]
        Si = sum([np.outer(v, v) for v in V])
        return mi, Si

    def reduceDimension(self, x):
        return np.dot(self.w, x)

    def classificate(self, x):
        return 0 if self.reduceDimension(x) > 0.5 * np.dot(self.w, self.m1 + self.m2) else 1

class MCFisher():
    W = []
    eigVal = []
    means = []

    def findW(self, data):
        data = [self.calculateClass(ci) for ci in data]
        self.means  = [val[0] for val in data]
        m = sum([val[0] * val[1] for val in data]) / sum([val[1] for val in data])
        Sw = sum([val[2] for val in data])
        Sb = sum([val[1] * np.outer(val[0] - m, val[0] - m) for val in data])
        self.eigVal, self.W = np.linalg.eig(np.dot(inv(Sw), Sb))

        idx = self.eigVal.argsort()[:: -1]   
        self.eigVal = self.eigVal[idx][0: -1]
        self.W = self.W[:, idx][0: -1]
        
        print('W: {0}'.format(self.W)) 
        print('Eigen values: {0}'.format(self.eigVal))

    def calculateClass(self, ci):
        nk = len(ci)
        ci = np.array(ci)
        mk = np.sum(ci, axis = 0) / nk
        V = [x - mk for x in ci]
        Sk = sum([np.outer(v, v) for v in V])
        return [mk, nk, Sk]

    def reduceDimension(self, x):
        y = np.dot(self.W, x)
        return y

    def reduceDimensionToClasses(self, data):
        reduced = []
        for ci in data:
            reduced.append([self.reduceDimension(x) for x in ci])
        return reduced