#!/usr/bin/python
import numpy as np
import math
from SyntheticValues import ClassificationValuesGenerator
from Classificators import LinearClassificator
from Plotter import plotClasses
import argparse

parser = argparse.ArgumentParser(description="Linear Classificator of K classes with D = 2.")
parser.add_argument("-k", action="store", dest="numberOfClasses", type=int, default=3,
                    help="Number of classses.")
parser.add_argument("-e", action="store", dest="testUsingNewData", type=bool, default=True,
                    help="Test de classifier using a different data set.")

if __name__ == "__main__":
    results = parser.parse_args()
    if results.numberOfClasses > 1:
        numberOfDataPerClass = np.random.uniform(80, 100, results.numberOfClasses)
        svg = ClassificationValuesGenerator(0, 30)
        trainingData, means = svg.getSyntheticValuesForClassification(numberOfDataPerClass, [[1, 0], [0, 1]])

        classificator = LinearClassificator()
        classificator.findW(trainingData)

        classificable = []
        classificated = [[] for i in range(0, results.numberOfClasses)]
        
        if not results.testUsingNewData:
            testData = trainingData
        else:
            testData, means = svg.getSyntheticValuesForClassificationWithMeans([50] * results.numberOfClasses, [[1, 0], [0, 1]], means)

        for i in xrange(results.numberOfClasses):
            for point in testData[i]:
                ci = classificator.classificate(point)
                classificated[ci].append(point)
                print("point {0} in class {1} must be {2}".format(point, ci, i))
        
        plotClasses(trainingData, classificated, "linearClassificationTest")
    else:
        raise ValueError("Number of classes must be greater than 1")