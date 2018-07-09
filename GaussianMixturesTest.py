#!/usr/bin/python
import numpy as np
import math
from SyntheticValues import ClassificationValuesGenerator
from KMeans import KMeans
from EM import EM
from Plotter import plotClasses
import argparse

parser = argparse.ArgumentParser(description="Mixture of Gaussians of K classes with D = 2.")
parser.add_argument("-k", action="store", dest="numberOfClasses", type=int, default=2,
                    help="Number of classses.")

if __name__ == "__main__":
    results = parser.parse_args()
    if results.numberOfClasses > 1:
        numberOfDataPerClass = np.random.uniform(80, 100, results.numberOfClasses)
        svg = ClassificationValuesGenerator(0, 30)
        classes, means = svg.getSyntheticValuesForClassification(numberOfDataPerClass, [[1, 0], [0, 1]], 2)
 
        classificable = []
        for cl in classes:
            classificable.extend(cl)

        classificator1 = EM()
        classificator1.expectationMaximization(classificable, results.numberOfClasses)

        classificator2 = KMeans()
        classificator2.calculateCentroids(classificable, results.numberOfClasses)
        print('Cluster 1: {0} - Cluster 2: {1}'.format(len(classificator1.clusters[0]), len(classificator1.clusters[1])))
        #print('Cluster EM: {0}'.format(classificator1.clusters))
        #print('Cluster K-Means: {0}'.format(classificator2.clusters))
        plotClasses(classificator1.clusters, [], "classification")
    else:
        raise ValueError("Number of classes must be greater than 1")