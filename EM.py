#!/usr/bin/python
import numpy as np
import math
import random as rand
from numpy.linalg import det
from numpy.linalg import inv

class EM:
    
    initialMeans = None
    means = []
    covs = []
    mixtures = []
    latents = None
    clusters = None
    totalIter = 0

    def expectationMaximization(self, data, K):
        for _ in xrange(K):
            self.means.append(rand.choice(data))
    
        [self.covs.append(np.identity(len(data[0]))) for _ in xrange(K)]
        self.mixtures = np.ones((K,))
        self.latents = np.zeros((len(data), K))

        maxIter = 30
        numIter = 0
        
        while numIter <= maxIter:
            self.expectation(data, K)
            self.maximitation(data, K)
            numIter += 1

        self.clusterData(data, K)

    def expectation(self, data, K):
        n = 0
        for x in data:
            for k in xrange(K):
                self.latents[n][k] = self.mixtures[k] * self.multivariateGaussian(x, k)
                if k == K - 1:
                    self.latents[n][k] /= sum(self.latents[n])
            n += 1 

    def maximitation(self, data, K):
        for k in xrange(K):
            n = 0
            Nk = 0
            self.means[k] = np.zeros((len(data[0]), ))
            for x in data:
                self.means[k] += self.latents[n][k] * np.array(x)
                Nk += self.latents[n][k]
                n += 1
            self.means[k] *= 1 / float(Nk)
            self.covs[k] = np.zeros((len(data[0]), len(data[0])))
            n = 0
            for x in data:
                v = x - self.means[k]
                self.covs[k] += self.latents[n][k] * np.outer(v, v)
                n += 1
            self.covs[k] *= 1 / float(Nk)
            self.mixtures[k] = Nk / float(len(data))
        
        #TODO: Calculate the log likelihood to use as condition     

    def multivariateGaussian(self, x, k):
        c = 1 / (2 * np.pi * det(self.covs[k])) ** 0.5
        v = np.array(x) - np.array(self.means[k])
        return c * np.exp(-0.5 * np.dot(v, np.dot(inv(self.covs[k]), v)))

    def clusterData(self, data, K):
        self.clusters = [[] for _ in xrange(K)]
        for x in data:
            i = np.argmax([self.multivariateGaussian(x, k) for k in xrange(K)])
            self.clusters[i].append(x)