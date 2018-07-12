#!/usr/bin/python
import numpy as np
import math
import random as rand
from numpy.linalg import det
from numpy.linalg import inv

maxLoglikelihoodDiff = 1

class EM:
    
    initialMeans = None
    means = []
    covs = []
    mixtures = []
    latents = None
    clusters = None
    totalIter = 0

    def expectationMaximization(self, data, K, initialMeans):
        if initialMeans != None:
            self.means = initialMeans
        else:
            for _ in xrange(K):
                self.means.append(rand.choice(data))
    
        [self.covs.append(np.identity(len(data[0]))) for _ in xrange(K)]
        self.mixtures = np.ones((K,))
        self.latents = np.zeros((len(data), K))

        maxIter = 15
        numIter = 0
        
        loglikelihoodOld = 0
        loglikelihoodNew = float('inf')
        while loglikelihoodNew - loglikelihoodOld > maxLoglikelihoodDiff and numIter <= maxIter:
            #print('Loglikelihood diff {0}'.format(loglikelihoodNew - loglikelihoodOld))
            loglikelihoodOld = loglikelihoodNew
            loglikelihoodNew = self.expectation(data, K)
            self.maximitation(data, K)
            numIter += 1

        self.clusterData(data, K)

    def expectation(self, data, K):
        loglikelihood = 0
        n = 0
        for x in data:
            loglikelihood_n = 0
            for k in xrange(K):
                val = self.mixtures[k] * self.multivariateGaussian(x, k)
                self.latents[n][k] = val
                loglikelihood_n += val                
                if k == K - 1:
                    self.latents[n][k] /= sum(self.latents[n])
            loglikelihood += np.log(loglikelihood_n)
            n += 1
        return loglikelihood

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
        c = 1 / np.sqrt((2 * np.pi * det(self.covs[k])))
        v = np.array(x) - np.array(self.means[k])
        return c * np.exp(-0.5 * np.dot(v, np.dot(inv(self.covs[k]), v)))

    def clusterData(self, data, K):
        self.clusters = [[] for _ in xrange(K)]
        for x in data:
            prob = [self.multivariateGaussian(x, k) for k in xrange(K)]
            #print('prob {0}'.format(prob))
            i = np.argmax(prob)
            self.clusters[i].append(x)