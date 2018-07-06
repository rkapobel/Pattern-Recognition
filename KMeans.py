#!/usr/bin/python
import numpy as np
from numpy.linalg import inv
import math
import random as rand

class KMeans:

    centroids = []
    clusters = None

    def calculateCentroids(self, data, K):
        for _ in xrange(K):
            self.centroids.append(rand.choice(data))
        
        clusters = self.updateClusters(data, K)
        numOfChanges = 0
        maxIter = 100
        iter = 0
        while numOfChanges / (K * 1.0) > 5 or iter <= maxIter:
            for i in xrange(K):
                self.centroids[i] = sum(clusters[i]) / (len(clusters[i]) * 1.0)
            iter += 1
            clusters, numOfChanges = self.updateClusters(data, K, clusters)

        self.clusters = clusters

    def updateClusters(self, data, K, oldClusters):
        clusters = [[] for _ in xrange(K)]
        [clusters[np.argmin([np.linalg.norm(x - mu, ord = 2) for mu in self.centroids])].append(x) for x in data]
        numOfChanges = sum([abs(len(clusters[i]) - len(oldClusters[i])) for i in xrange(K))])

        return clusters, numOfChanges