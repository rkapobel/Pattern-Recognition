#!/usr/bin/python
import numpy as np
import math
from inspect import isfunction, ismethod, isbuiltin

class TestValues(object):
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
        
    def getNewValues(self):
        noises = np.random.normal(self.mu, self.sigma, self.N)
        func = lambda x, e: self.f(x) + e
        xData = np.random.uniform(self.a, self.b, self.N)
        yData = [func(x, e) for x, e in zip(xData, noises)]
        return (list(xData), yData)