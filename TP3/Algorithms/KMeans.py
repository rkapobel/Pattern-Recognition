#!/usr/bin/python
import numpy as np
import math
import random as rand
from numpy.linalg import norm

class KMeans:

    maxNumOfChangesAllowed = 2
    initialCentroids = None
    centroids = []
    clusters = None
    totalIter = 0

    def calculateCentroids(self, data, K):
        for _ in xrange(K):
            self.centroids.append(rand.choice(data))
        
        self.initialCentroids = self.centroids
        clusters = self.clusterData(data, K, [[] for _ in xrange(K)])
        
        numOfChanges = 0
        oldNumOfChanges = float('inf')
        maxIter = 100
        numIter = 0
        
        while abs(numOfChanges - oldNumOfChanges) >= self.maxNumOfChangesAllowed or numIter <= maxIter:
            for i in xrange(K):
                self.centroids[i] = sum(clusters[i]) / (len(clusters[i]) * 1.0)
            numIter += 1
            oldClusters = clusters
            clusters = self.clusterData(data, K, clusters)
            oldNumOfChanges = numOfChanges
            numOfChanges = sum([abs(len(clusters[i]) - len(oldClusters[i])) for i in xrange(K)])

        self.totalIter = numIter
        self.clusters = clusters

    def clusterData(self, data, K, oldClusters):
        clusters = [[] for _ in xrange(K)]
        [clusters[np.argmin([norm(np.array(x) - np.array(mu), ord = 2) for mu in self.centroids])].append(np.array(x)) for x in data]

        return clusters