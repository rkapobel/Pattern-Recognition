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
        numberOfDataPerClass = np.random.uniform(10, 15, results.numberOfClasses)
        svg = ClassificationValuesGenerator(0, 10, 0, 10)
        classes = svg.getSyntheticValuesForClassification(numberOfDataPerClass)
        classificator = Classificator()
        classificator.findW(classes)
        # The way to use ClassificationValuesGenerator is a little dirty
        classificable = svg.getSyntheticValuesForClassification([30])
        print(classificable)
        classificated = [[]] * results.numberOfClasses
        for point in zip(classificable[0][0], classificable[0][1]):
            cl = classificator.classificate(point[0], point[1])
            classificated[cl].append(point)
            #print("point {0} in class {1}".format(point, cl))
        plotClasses(classes, classificated, "classification")
    else:
        raise ValueError("Number of classes must be greater than 1")