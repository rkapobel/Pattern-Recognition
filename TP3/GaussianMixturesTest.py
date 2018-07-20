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
parser.add_argument("-t", action="store", dest="testsToRun", type=int, default=0,
                    help="0: run EM and K-Means. 1: Will run only EM. 2: Will run only K-Means.")
parser.add_argument("-em", action="store", dest="numberOfIterationsOfEM", type=int, default=2000,
                    help="Number of iterations of EM.")
parser.add_argument("-km", action="store", dest="numberOfIterationsOfKMeans", type=int, default=200,
                    help="Number of iterations of K-Means.")

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

        if results.testsToRun == 0 or results.testsToRun == 1:
            classificator1 = EM()
            classificator1.expectationMaximization(classificable, results.numberOfClasses, results.numberOfIterationsOfEM)
            plotClasses(trainingData, classificator1.clusters, "classificationEM")
         
        if results.testsToRun == 0 or results.testsToRun == 2:
            classificator2 = KMeans()
            classificator2.calculateCentroids(classificable, results.numberOfClasses, results.numberOfIterationsOfKMeans)
            plotClasses(trainingData, classificator2.clusters, "classificationK-Means")
    else:
        raise ValueError("Number of classes must be greater than 1")