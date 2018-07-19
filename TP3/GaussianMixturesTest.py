#!/usr/bin/python
import numpy as np
import math
from SyntheticValues import ClassificationValuesGenerator
from Algorithms.KMeans import KMeans
from Algorithms.EM import EM
from Plotter import plotClasses, plotConvergence
import argparse
from random import shuffle

parser = argparse.ArgumentParser(description="Mixture of Gaussians of K classes with D = 2.")
parser.add_argument("-k", action="store", dest="numberOfClasses", type=int, default=2,
                    help="Number of classses.")

if __name__ == "__main__":
    results = parser.parse_args()
    if results.numberOfClasses > 1:
        numberOfDataPerClass = np.random.uniform(80, 100, results.numberOfClasses)
        svg = ClassificationValuesGenerator(0, 10)
        values = svg.getSyntheticValuesForClassification(numberOfDataPerClass)
 
        trainingData = values[0]
        cov = values[1]
        means = values[2]
 
        classificable = []
        for cl in trainingData:
            classificable.extend(cl)

        shuffle(classificable)

        classificator1 = EM()
        classificator1.expectationMaximization(classificable, results.numberOfClasses, means)

        classificator2 = KMeans()
        classificator2.calculateCentroids(classificable, results.numberOfClasses)
        
        #print('Cluster EM: {0}'.format(classificator1.clusters))
        #print('Cluster K-Means: {0}'.format(classificator2.clusters))

        plotClasses(trainingData, classificator1.clusters, "classificationEM")
        plotConvergence(classificator1.getEpochs(), classificator1.likelihoods, "classificationEMLikelihoods", "Likelihood")
        plotClasses(trainingData, classificator2.clusters, "classificationK-Means")
        plotConvergence(classificator2.getEpochs(), classificator2.objetives, "classificationK-MeansObjetives", "Objetive")
    else:
        raise ValueError("Number of classes must be greater than 1")