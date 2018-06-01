#!/usr/bin/python
import numpy as np
import math
from SyntheticValues import ClassificationValuesGenerator
from Classificator import Classificator
from Plotter import plotClasses
import argparse

parser = argparse.ArgumentParser(description="Classificator of K classes with D = 2.")
parser.add_argument("-k", action="store", dest="numberOfClasses", type=int, default=3,
                    help="Number of classses.")

if __name__ == "__main__":
    results = parser.parse_args()
    if results.numberOfClasses > 1:
        numberOfDataPerClass = np.random.uniform(50, 100, results.numberOfClasses)
        svg = ClassificationValuesGenerator(0, 30, 0, 30)
        classes, means = svg.getSyntheticValuesForClassification(numberOfDataPerClass, [[1, 0], [0, 1]])

        classificator = Classificator()
        classificator.findW(classes)

        # The way to use ClassificationValuesGenerator is a little dirty
        classificable, means = svg.getSyntheticValuesForClassificationWithMeans([50] * results.numberOfClasses, [[6, 2], [1, 5]], means)
        classificated = [[] for i in range(0, results.numberOfClasses)]
        for i in xrange(results.numberOfClasses):
            for point in zip(classificable[i][0], classificable[i][1]):
                cl = classificator.classificate(point[0], point[1])
                classificated[cl].append(point)
                print("point {0} in class {1} must be {2}".format(point, cl, i))
        plotClasses(classes, classificated, "classification")
    else:
        raise ValueError("Number of classes must be greater than 1")