#!/usr/bin/python
import numpy as np
import math
from SinValuesGenerator import SinValuesGenerator
from PolinomialRegression import PolynomialRegression
from ErrorPlotter import plotErrorsByDegree, plotErrorsByLogLambda

if __name__ == "__main__":
    N = 10
    svg = SinValuesGenerator(2*math.pi, 0, 1, N, 0, 1)
    trngData = svg.sinValues()
    testData = svg.sinValues()
    
    trainingErrors = []
    testErrors = []
    """
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
