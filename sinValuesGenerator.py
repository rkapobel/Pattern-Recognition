#!/usr/bin/python

import numpy as np
from numpy.linalg import inv
import math
import matplotlib.pyplot as plot

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
        return (list(xData), yData)

# The next classes will be moved to new files when possible.

class PolynomialRegression:
    reg = 0
    M = 0
    w = None

    def __init__(self, reg, M):
        self.reg = reg
        self.M = M

    def findW(self, data):
        self.checkData(data)            
        t = np.array(data[1])
        X = self.getX(data[0])
        Xt = np.transpose(X)
        XtX = np.dot(Xt, X) + (np.identity(X.shape[1]) * self.reg if self.reg > 0 else 0)
        Xt_t = np.dot(Xt, t)
        try:
            XtXINv = inv(XtX)
            self.w = np.dot(XtXINv, Xt_t)
        except np.linalg.LinAlgError as e:
            print(e)

    def y(self, x):
        self.checkW()
        return np.dot([x ** i for i in xrange(self.M + 1)], self.w)

    def error(self, data, checkData = True):
        # Mmmmmmmh
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
        e = sum([(xi - ti) ** 2 for xi, ti in zip(X_dot_w, t)])
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

# The scritps to make the plots and the scripts to make the analysis will be separated in different modules.

def plotErrorsByDegree(degrees, errors):
    plot.plot(degrees, errors[0], "b-o", label="Training")
    plot.plot(degrees, errors[1], "r-o", label="Test")
    plot.legend(loc='upper right', shadow=True)
    plot.axes().set_title("$Degree$ $influence$")
    plot.axes().set_xlabel("$M$")
    plot.axes().set_ylabel("$E_{drms}$")
    plot.show()

def plotErrorsByLogLambda(lambdas, errors):
    plot.plot(lambdas, errors[0], "b-o", label="Training")
    plot.plot(lambdas, errors[1], "r-o", label="Test")
    plot.legend(loc='upper right', shadow=True)
    plot.axes().set_title("$\lambda$ $influence$")
    plot.axes().set_xlabel("$log(\lambda)$")
    plot.axes().set_ylabel("$E_{rms}$")
    plot.show()

if __name__ == "__main__":
    N = 10
    svg = SinValuesGenerator(2*math.pi, 0, 1, N, 0, 1)
    trngData = svg.sinValues()
    testData = svg.sinValues()
    
    trainingErrors = []
    testErrors = []
    for degree in xrange(N):
        pr = PolynomialRegression(0, degree)
        pr.findW(trngData) # Finding w* for training data
        print(pr.w)
        trngError = pr.ems(trngData)
        print("Degree: " + str(degree))
        print("trng error: " + str(trngError))
        trainingErrors.append(np.log(trngError))
        testError = pr.ems(testData)
        print("test error: " + str(testError))
        testErrors.append(np.log(testError))

    plotErrorsByDegree(list(range(N)), (trainingErrors, testErrors))
    """
    trainingErrors = []
    testErrors = []
    for l in xrange(-40, 1):
        l = math.exp(l)
        pr = PolynomialRegression(l, N-1)
        pr.findW(trngData)
        trngError = pr.ems(trngData)
        print("log(lambda): " + str(l))
        print("trng error: " + str(trngError))
        trainingErrors.append(np.log(trngError))
        testError = pr.ems(testData)
        print("test error: " + str(testError))
        testErrors.append(np.log(testError))

    plotErrorsByLogLambda(list(range(-40, 1)), (trainingErrors, testErrors))
    """