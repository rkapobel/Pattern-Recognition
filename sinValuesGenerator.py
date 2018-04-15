#!/usr/bin/python

import numpy as np
from numpy.linalg import inv
import math

class SinValuesGenerator:
    multiplier = 1
    mu = 0
    sigma = 1
    N = 0
    a = 0
    b = 1

    def __init__(self, multiplier, mu, sigma, N, a, b):
        self.multiplier = multiplier
        self.mu = mu
        self.sigma = sigma
        self.N = N
        self.a = a
        self.b = b

    def sinValues(self):
        noises = np.random.normal(self.mu, self.sigma, self.N)
        func = lambda x, e: math.sin(self.multiplier * x) + e
        data = [func(x, e) for x, e in zip(np.random.uniform(self.a, self.b, self.N), noises)]
        return data

# The next classes will be moved to new files when possible.

class PolinomialRegression:
    reg = 0
    M = 1
    X = None
    t = None
    xData = None
    yData = None
    w = None

    def __init__(self, reg, M):
        self.reg = reg
        self.M = M

    def findW(self, xData, yData):
        if xData is None or yData is None or len(xData) == 0 or len(xData) != len(yData):
            raise ValueError("xData and yData must be not zero equal size.")
            
        self.xData = xData.insert(1, 0)
        self.yData = yData
        self.t = np.array(self.yData)
        func = lambda x, j: x ** j
        self.X = np.array([[func(x, j) for j in xrange(self.M+1)] for x in self.xData])
        Xt = np.transpose(self.X)
        XtX = np.dot(np.dot(Xt, self.X) + np.identity(len(xData) + 1) * self.reg)
        Xt_t = np.dot(Xt, self.t)
        try:
            XtXINv = inv(XtX)
            self.w = np.dot(XtXINv, Xt_t)
        except np.linalg.LinAlgError as e:
            print(e)

    def error(self):
        if self.w is None:
            raise ValueError("Error is inf. You did not calculate the error yet.")
            
        reg_cond = reduce(lambda tot, wi: tot + (wi ** 2), self.w)
        X_dot_w = np.dot(self.X, self.w)
        # equals to map(lambda xi, ti: (xi - ti) ** 2, zip(X_dot_w, self.t))
        e = sum([(xi - ti) ** 2 for xi, ti in zip(X_dot_w, self.t)])
        return 0.5 * (e + self.reg * reg_cond)

    def ems(self):
        return math.sqrt((2 * self.error()) / self.t.shape[0])

if __name__ == "__main__":
    svg = SinValuesGenerator(2*math.pi, 0, 1, 10, 0, 10)
    data = svg.sinValues()
    print(data)