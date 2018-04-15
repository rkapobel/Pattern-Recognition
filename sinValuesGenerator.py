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
        xData = np.random.uniform(self.a, self.b, self.N)
        yData = [func(x, e) for x, e in zip(xData, noises)]
        return (xData, yData)

# The next classes will be moved to new files when possible.

class PolynomialRegression:
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

    def findW(self, data):
        if isinstance(data, tuple) and data[0] is None or data[1] is None or len(data[0]) == 0 or len(data[0]) != len(data[1]):
            raise ValueError("data must be a tuple and [0] and [1] must be not zero arrays with equal size.")
            
        self.xData = data[0]
        self.yData = data[1]
        self.t = np.array(self.yData)
        func = lambda x, j: x ** j
        self.X = np.array([[func(x, j) for j in xrange(self.M + 1)] for x in self.xData])
        Xt = np.transpose(self.X)
        print(len(self.xData))
        print(self.X.shape)
        XtX = np.dot(Xt, self.X) + np.identity(self.X.shape[1]) * self.reg
        Xt_t = np.dot(Xt, self.t)
        try:
            XtXINv = inv(XtX)
            self.w = np.dot(XtXINv, Xt_t)
        except np.linalg.LinAlgError as e:
            print(e)

    def error(self):
        self.checkW()    
        reg_cond = reduce(lambda tot, wi: tot + (wi ** 2), self.w)
        X_dot_w = np.dot(self.X, self.w)
        # equals to map(lambda xi, ti: (xi - ti) ** 2, zip(X_dot_w, self.t))
        e = sum([(xi - ti) ** 2 for xi, ti in zip(X_dot_w, self.t)])
        return 0.5 * (e + self.reg * reg_cond)

    def ems(self):
        return math.sqrt((2 * self.error()) / self.t.shape[0])

    def y(self, x):
        self.checkW()
        return np.dot([x ** i for i in xrange(self.M + 1)], self.w)

    def checkW(self):
        if self.w is None:
            raise ValueError("Error is inf. You did not calculate the error yet.")

if __name__ == "__main__":
    svg = SinValuesGenerator(2*math.pi, 0, 1, 50, 0, 10)
    data1 = svg.sinValues()
    data2 = svg.sinValues()
    pr = PolynomialRegression(0, 5)
    pr.findW(data1)
    print(pr.w)
    
    print(data1)
    print(data2)
