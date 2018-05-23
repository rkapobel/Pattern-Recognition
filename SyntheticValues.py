#!/usr/bin/python
import numpy as np
import math
from inspect import isfunction, ismethod, isbuiltin

class RegressionValuesGenerator(object):
    mu = 0
    sigma = 1
    N = 0
    a = 0
    b = 1

    def __init__(self, mu, sigma, N, f, a, b):
        self.mu = mu
        self.sigma = sigma
        self.N = N
        self.f = f
        self.a = a
        self.b = b

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, value):
        if isfunction(value) == False and ismethod(value) == False and isbuiltin(value) == False:
            raise Exception("You must pass a mathematical function or method for attribute f")
        self._f = value
        
    def getSyntheticValuesForRegression(self):
        noises = np.random.normal(self.mu, self.sigma, self.N)
        xData = np.random.uniform(self.a, self.b, self.N)
        #func = lambda x, e: self.f(x) + e
        #yData = [func(x, e) for x, e in zip(xData, noises)]
        yData = self.f(xData) + noises
        return (list(xData), yData)

class ClassificationValuesGenerator(object):
    x1Start = 0
    x1End = 0
    x2Start = 0
    x2End = 0
    numberOfPointsPerClass = 0

    def __init__(self, x1Start, x1End, x2Start, x2End, numberOfPointsPerClass):        
        if len(numberOfPointsPerClass) < 2:
            raise ValueError("Must be at least two classes.")

        self.x1Start = x1Start
        self.x1End = x1End
        self.x2Start = x2Start
        self.x2End = x2End
        self.numberOfPointsPerClass = numberOfPointsPerClass
        

    def getSyntheticValuesForClassification(self):
        #Figure out a way to control de cov matrix per class
        classes = []

        K = len(self.numberOfPointsPerClass)

        x1Means = np.random.uniform(self.x1Start, self.x1End, K)
        x2Means = np.random.uniform(self.x1Start, self.x2End, K)

        for i in xrange(K):
            means = [x1Means[i], x2Means[i]]
            cov = [[1, 0], [0, 1]]
            x1, x2 = np.random.multivariate_normal(means, cov, self.numberOfPointsPerClass[i]).T
            classes.append((x1, x2))
