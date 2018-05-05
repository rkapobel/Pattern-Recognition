#!/usr/bin/python
import numpy as np
import math
from TestValues import TestValues
from PolinomialRegression import PolynomialRegression
from ErrorPlotter import plotErrorsByDegree, plotErrorsByLogLambda, plotOriginalVsEstimated

def sin(x):
    return math.sin(2 * math.pi * x)

def log(x):
    return math.log(2 * math.pi * (x + 0.000001))

def pol(x):
    return -3*(x**3) + 3*(x**2) + 4*x

if __name__ == "__main__":
    N = 10
    function = sin
    svg = TestValues(0, 1, N, function, 0, 1)
    trngData = svg.getNewValues()
    testData = svg.getNewValues()
    
    trainingErrors = []
    testErrors = []
    
    for degree in xrange(N):
        pr = PolynomialRegression(0, degree)
        pr.findW(trngData) # Finding w* for training data
        trngError = pr.ems(trngData)
        print("Degree: " + str(degree))
        print("trng error: " + str(trngError))
        trainingErrors.append(trngError)
        testError = pr.ems(testData)
        print("test error: " + str(testError))
        testErrors.append(testError)
        if degree == 3:
            plotOriginalVsEstimated(function, pr.y, np.linspace(0, 1, 1000))

    #plotErrorsByDegree(list(range(N)), (trainingErrors, testErrors))
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
        trainingErrors.append(trngError)
        testError = pr.ems(testData)
        print("test error: " + str(testError))
        testErrors.append(testError)

    plotErrorsByLogLambda(list(range(-40, 1)), (trainingErrors, testErrors))
    """