#!/usr/bin/python
import numpy as np
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
        return (list(xData), yData)

# The scritps to make the plots and the scripts to make the analysis will be separated in different modules.