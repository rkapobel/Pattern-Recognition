#!/usr/bin/python
import numpy as np
import math
import random as rand
from numpy.linalg import norm

def chunks(toSplit, n):
    splitted = []
    for i in range(0, len(toSplit), n):
        splitted.append(toSplit[i:i + n])
    return splitted

class KMeans:

    maxIter = 100
    maxNumOfChangesAllowed = 2
    initialCentroids = None
    centroids = None
    clusters = None
    totalIter = 0
    objetives = []

    def getEpochs(self):
        return list(range(self.totalIter))

    def calculateCentroids(self, data, K, maxIter = 100):
        sampleClusters = chunks(data, len(data) / K)
        self.centroids = [sum(np.array(cluster)) / float(len(cluster)) for cluster in sampleClusters]
        
        self.initialCentroids = self.centroids
        clusters = self.clusterData(data, K)
        
        numOfChanges = 0
        oldNumOfChanges = float('inf')
        self.maxIter = maxIter
        numIter = 0
        
        while abs(numOfChanges - oldNumOfChanges) >= self.maxNumOfChangesAllowed and numIter <= self.maxIter:
            for i in xrange(K):
                self.centroids[i] = sum(clusters[i]) / float(len(clusters[i]))
            numIter += 1
            oldClusters = clusters
            clusters = self.clusterData(data, K)
            self.calculateNewObjetive(clusters)
            oldNumOfChanges = numOfChanges
            numOfChanges = sum([abs(len(clusters[i]) - len(oldClusters[i])) for i in xrange(K)])

        self.totalIter = numIter
        self.clusters = clusters

    def clusterData(self, data, K):
        clusters = [[] for _ in xrange(K)]
        [clusters[np.argmin([norm(np.array(x) - np.array(mu), ord = 2) for mu in self.centroids])].append(np.array(x)) for x in data]

        return clusters

    def calculateNewObjetive(self, clusters):
        self.objetives.append(sum([sum([sum([norm(np.array(x) - np.array(mu), ord = 2) for mu in self.centroids]) for x in cluster]) for cluster in clusters]))