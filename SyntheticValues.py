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
        np.random.seed(10000)

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
    a = 0
    b = 0

    def __init__(self, a, b):        
        self.a = a
        self.b = b
        np.random.seed(10000)

    def getEllipticValuesForClassification(self):
        numberOfDataPerClass = np.random.uniform(80, 100, 2)
        means = [np.random.uniform(self.a, self.b, 1) for _ in xrange(2)]
        means = map(list, zip(*means))
        classes1, means1 = self.getSyntheticValuesForClassificationWithMeans(numberOfDataPerClass[0], [[1, 0], [0, 1]], means)
        classes2, means2 = self.getSyntheticValuesForClassificationWithMeans(numberOfDataPerClass[1], [[3, 1], [0, 10]], means)
        return [classes1[0], classes2[0]], [means1[0], means2[0]]

    def getSyntheticValuesForClassification(self, numberOfPointsPerClass, cov, dim = 2): 
        K = len(numberOfPointsPerClass)
        means = [np.random.uniform(self.a, self.b, K) for _ in xrange(dim)]
        means = map(list, zip(*means))
        return self.getSyntheticValuesForClassificationWithMeans(numberOfPointsPerClass, cov, means)

    def getSyntheticValuesForClassificationWithMeans(self, numberOfPointsPerClass, cov, means):
        classes = []

        K = len(numberOfPointsPerClass)

        for i in xrange(K):
            X = np.random.multivariate_normal(means[i], cov, int(numberOfPointsPerClass[i])).T
            classes.append(map(list, zip(*X)))

        return classes, means