#!/usr/bin/python
import numpy as np
import math
from TestValues import TestValues
from PolynomialRegression import PolynomialRegression
from ErrorPlotter import plotErrorsByDegree, plotErrorsByLogLambda, plotOriginalVsEstimated
import argparse

def sin(x):
    return np.sin(2 * math.pi * x)

def log(x):
    return np.log(2 * math.pi * (x + 0.000001))

def pol(x):
    return -3*(x**3) + 3*(x**2) + 4*x

functions = [sin, log, pol]

parser = argparse.ArgumentParser(description='Polynomial Regression Tests.')
parser.add_argument('-t', action='store', dest='test',
                    help='a: Influence of the degree hyperparameter. \n b: Influence of the lambda hyperparameter.')
parser.add_argument('-n', action='store', dest='N', default=10,
                    help='Number of data to generate. Test A will run for degree M: 0..N-1.')
parser.add_argument('-f', action='store', dest='function', type=str, default='sin',
                    help='Function to test:' + str([f.__name__ for f in functions]))

if __name__ == "__main__":
    results = parser.parse_args()
    print("Running test: " + str(results.test))

    N = results.N
    function = filter(lambda function: function.__name__ == results.function, functions)[0]
    svg = TestValues(0, 1, N, function, 0, 1)
    trngData = svg.getNewValues()
    testData = svg.getNewValues()

    if results.test == "a":
        trainingErrors = []
        testErrors = []
        
        for degree in xrange(N):
            pr = PolynomialRegression(0, degree)
            pr.findW(trngData) # Finding w* for training data
            trngError = pr.ems(trngData)
            trainingErrors.append(trngError)
            testError = pr.ems(testData)
            testErrors.append(testError)
            print("Degree: " + str(degree))
            print("trng error: " + str(trngError))
            print("test error: " + str(testError))
            if degree == 3:
                plotOriginalVsEstimated(function, pr.y, np.linspace(0, 1, 1000), results.function, degree, 0)

        plotErrorsByDegree(list(range(N)), (trainingErrors, testErrors))        
    else:
        trainingErrors = []
        testErrors = []
        for l in xrange(-40, 1):
            l = math.exp(l)
            pr = PolynomialRegression(l, N-1)
            pr.findW(trngData)
            trngError = pr.ems(trngData)
            trainingErrors.append(trngError)
            testError = pr.ems(testData)
            testErrors.append(testError)
            print("log(lambda): " + str(l))
            print("trng error: " + str(trngError))
            print("test error: " + str(testError))
            
        plotErrorsByLogLambda(list(range(-40, 1)), (trainingErrors, testErrors))
