#!/usr/bin/python
import numpy as np
import math
import random as rand
from numpy.linalg import det
from numpy.linalg import inv

class EM:
   
    means = []
    covs = []
    mixtures = []
    likelihoods = []
    latents = None
    clusters = None
    totalIter = 0

    def getEpochs(self):
        return list(range(self.totalIter))

    def expectationMaximization(self, data, K, maxIter = 2000):
        for _ in xrange(K):
            self.means.append(rand.choice(data))
        
        [self.covs.append(np.identity(len(data[0]))) for _ in xrange(K)]
        self.mixtures = np.ones((K,)) * (1 / float(K))
        self.latents = np.zeros((len(data), K))
        
        self.maxIter = maxIter
        numIter = 0
       
        while numIter <= self.maxIter:
            self.clusterData(data, K)
            self.likelihoods.append(self.expectation(data, K))
            self.maximitation(data, K)
            covsOK = True
            for i in xrange(K):
                try:
                    inv(self.covs[i])
                except np.linalg.LinAlgError as e:
                    print(e)
                    covsOK = False
                    break
                if det(self.covs[i]) < 0:
                    covsOK = False
                    break
            if not covsOK:
                break
            numIter += 1

        self.totalIter = numIter

    def expectation(self, data, K):
        loglikelihood = 0
        n = 0
        for x in data:
            loglikelihood_n = 0
            for k in xrange(K):
                val = self.mixtures[k] * self.multivariateGaussian(x, k, K)
                self.latents[n][k] = val
                loglikelihood_n += val
            self.latents[n] /= float(sum(self.latents[n]))
            loglikelihood += math.log(loglikelihood_n)
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
        
    def multivariateGaussian(self, x, k, K):
        c = 1 / math.sqrt(((2 * math.pi)**K) *  det(self.covs[k]))
        v = np.array(x) - np.array(self.means[k])
        return c * math.exp(-0.5 * np.dot(v, np.dot(inv(self.covs[k]), v)))

    def clusterData(self, data, K):
        self.clusters = [[] for _ in xrange(K)]
        for x in data:
            prob = [self.multivariateGaussian(x, k, K) for k in xrange(K)]
            i = np.argmax(prob)
            self.clusters[i].append(x)