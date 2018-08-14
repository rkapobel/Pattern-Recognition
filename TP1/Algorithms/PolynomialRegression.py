#!/usr/bin/python
import numpy as np
from numpy.linalg import inv
import math

class PolynomialRegression:
   
    def __init__(self, reg, M):
        self.w = None
        self.reg = reg
        self.M = M

    def findW(self, data):
        self.checkData(data)            
        t = np.array(data[1])
        X = self.getX(data[0])
        #Xt = np.transpose(X)
        Xt = X.T
        XtX = np.dot(Xt, X) + (np.identity(X.shape[1]) * self.reg if self.reg > 0 else 0)
        Xt_t = np.dot(Xt, t)
        try:
            XtXINv = inv(XtX)
            self.w = np.dot(XtXINv, Xt_t)
        except np.linalg.LinAlgError as e:
            print(e)

    def y(self, x):
        self.checkW()
        #return np.dot([x ** i for i in xrange(self.M + 1)], self.w)
        return np.polynomial.polynomial.polyval(x, self.w)

    def error(self, data, checkData = True):
        if checkData == True:
            self.checkData(data)
        self.checkW()
        t = np.array(data[1])
        X = self.getX(data[0])
        reg_cond = 0
        if self.reg > 0:
            reg_cond = reduce(lambda tot, wi: tot + (wi ** 2), self.w)
        X_dot_w = np.dot(X, self.w)
        # equals to map(lambda xi, ti: (xi - ti) ** 2, zip(X_dot_w, self.t))
        #e = sum([(xi - ti) ** 2 for xi, ti in zip(X_dot_w, t)])
        e = sum((X_dot_w - t) ** 2)
        return 0.5 * (e + self.reg * reg_cond)

    def ems(self, data):
        self.checkData(data)
        try:
            return math.sqrt(2 * self.error(data, False) / float(len(data[0])))
        except ZeroDivisionError as e:
            print("T is a void vector:\n" + str(e))

    def getX(self, data):
        func = lambda x, j: x ** j
        return np.array([[func(x, j) for j in xrange(self.M + 1)] for x in data])

    def checkData(self, data):
        if isinstance(data, tuple) and data[0] is None or data[1] is None or len(data[0]) == 0 or len(data[0]) != len(data[1]):
            raise ValueError("data must be a tuple and [0] and [1] must be not zero arrays with equal size.")

    def checkW(self):
        if self.w is None:
            raise ValueError("Error is inf. You did not calculate the error yet.")