#!/usr/bin/python
import numpy as np
import math
import random as rand
from numpy.linalg import det
from numpy.linalg import inv

maxLoglikelihoodDiff = 0.05

def chunks(toSplit, n):
    splitted = []
    for i in range(0, len(toSplit), n):
        splitted.append(toSplit[i:i + n])
    return splitted

class EM:
    
    maxIter = 100
    initialMeans = None
    means = []
    covs = []
    mixtures = []
    likelihoods = []
    latents = None
    clusters = None
    totalIter = 0

    def getEpochs(self):
        return list(range(self.totalIter))

    def expectationMaximization(self, data, K, maxIter = 100):
        sampleClusters = chunks(data, len(data) / K)
        for cluster in sampleClusters:
            cluster = np.array(cluster)
            self.means.append(sum(cluster) / float(len(cluster)))
            self.covs.append(np.cov(cluster.T))
            self.mixtures.append(cluster.shape[0] / float(len(data)))
        self.mixtures = np.array(self.mixtures)
        self.latents = np.zeros((len(data), K))

        self.maxIter = maxIter
        numIter = 0
        
        loglikelihoodOld = 0
        loglikelihoodNew = float('inf')

        #loglikelihoodNew - loglikelihoodOld > maxLoglikelihoodDiff and  
        while numIter <= self.maxIter:
            #print('Loglikelihood diff {0}'.format(loglikelihoodNew - loglikelihoodOld))
            loglikelihoodOld = loglikelihoodNew
            try:
                loglikelihoodNew = self.expectation(data, K)
                self.likelihoods.append(loglikelihoodNew)
                self.maximitation(data, K)
                self.clusterData(data, K)
                numIter += 1
            except np.linalg.LinAlgError as e:
                print(e)
                break

        self.totalIter = numIter

    def expectation(self, data, K):
        loglikelihood = 0
        n = 0
        for x in data:
            loglikelihood_n = 0
            for k in xrange(K):
                val = self.mixtures[k] * self.multivariateGaussian(x, k)
                self.latents[n][k] = val
                loglikelihood_n += val                
            self.latents[n][k] /= sum(self.latents[n])
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
        
    def multivariateGaussian(self, x, k):
        d = det(self.covs[k])
        print('det is: {0}'.format(d))
        squareRoot = math.sqrt((2 * math.pi *  abs(d)))
        print('sqrt is: {0}'.format(squareRoot))
        c = 1 / squareRoot
        print('c is: {0}'.format(c))
        v = np.array(x) - np.array(self.means[k])
        return c * math.exp(-0.5 * np.dot(v, np.dot(inv(self.covs[k]), v)))

    def clusterData(self, data, K):
        self.clusters = [[] for _ in xrange(K)]
        for x in data:
            prob = [self.multivariateGaussian(x, k) for k in xrange(K)]
            i = np.argmax(prob)
            self.clusters[i].append(x)