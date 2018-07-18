#!/usr/bin/python
import numpy as np
import math
from SyntheticValues import ClassificationValuesGenerator
from Algorithms.Classificators import LinearClassificator
from Plotter import plotClasses
import argparse

parser = argparse.ArgumentParser(description="Linear Classificator of K classes with D = 2.")
parser.add_argument("-k", action="store", dest="numberOfClasses", type=int, default=3,
                    help="Number of classses.")
parser.add_argument("-e", action="store", dest="testUsingTrainingData", type=int, default=1,
                    help="1: Test de classifier using a different data set. 0: Test using the training data set.")

if __name__ == "__main__":
    results = parser.parse_args()
    if results.numberOfClasses > 1:
        numberOfDataPerClass = np.random.uniform(80, 100, results.numberOfClasses)
        svg = ClassificationValuesGenerator(0, 30)
        values = svg.getSyntheticValuesForClassification(numberOfDataPerClass)

        trainingData = values[0]
        cov = values[1]
        means = values[2]

        classificator = LinearClassificator()
        classificator.findW(trainingData)

        classificable = []
        classificated = [[] for i in range(0, results.numberOfClasses)]
        
        if results.testUsingTrainingData == 0:
            testData = trainingData
        else:
            testData = svg.getSyntheticValuesForClassificationWithMeans([50] * results.numberOfClasses, cov, means)

        for i in xrange(results.numberOfClasses):
            for point in testData[i]:
                ci = classificator.classificate(point)
                classificated[ci].append(point)
                print("point {0} in class {1} must be {2}".format(point, ci, i))
        
        plotClasses(trainingData, classificated, "linearClassificationTest")
    else:
        raise ValueError("Number of classes must be greater than 1")