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

    def getEllipticValues(self, circle_r, numValues):
        # center of the circle (x, y)
        circle_x = 5
        circle_y = 7

        values = []
        mean = [circle_x, circle_y]

        for _ in xrange(int(numValues)):
            # random angle
            alpha = 2 * math.pi * np.random.random()
            # random radius
            r = circle_r * np.random.uniform(1, 5, 1)
            # calculating coordinates
            x = r * math.cos(alpha) + circle_x
            y = r * math.sin(alpha) + circle_y
            values.append([x[0], y[0]])

        return values, mean

    def getEllipticValuesForClassification(self):
        numberOfDataPerClass = np.random.uniform(80, 100, 2)

        class1, mean1 = self.getEllipticValues(10, numberOfDataPerClass[0])
        class2, mean2 = self.getEllipticValues(90, numberOfDataPerClass[1])

        return [class1, class2], [mean1, mean2]

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