#!/usr/bin/python
import numpy as np
import math
import random as rand
from numpy.linalg import norm

class KMeans:

    centroids = []
    clusters = None
    totalIter = 0
    objetives = []

    def getEpochs(self):
        return list(range(self.totalIter))

    def calculateCentroids(self, data, K, maxIter = 200):
        for _ in xrange(K):
            self.centroids.append(rand.choice(data))
        self.clusterData(data, K)
        self.maxIter = maxIter
        numIter = 0
         
        while numIter <= self.maxIter:
            for i in xrange(K):
                self.centroids[i] = sum(self.clusters[i]) / float(len(self.clusters[i]))
            numIter += 1
            self.clusterData(data, K)
            self.calculateNewObjetive()

        self.totalIter = numIter

    def clusterData(self, data, K):
        self.clusters = [[] for _ in xrange(K)]
        [self.clusters[np.argmin([norm(np.array(x) - np.array(mu), ord = 2)**2 for mu in self.centroids])].append(np.array(x)) for x in data]

    def calculateNewObjetive(self):
        self.objetives.append(sum([sum([sum([(norm(np.array(x) - np.array(mu), ord = 2))**2 for mu in self.centroids]) for x in cluster]) for cluster in self.clusters]))