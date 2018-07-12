#!/usr/bin/python
import numpy as np
import math
from SyntheticValues import ClassificationValuesGenerator
from LogisticRegression import LogisticRegression, MCLogisticRegression
from Plotter import plotClasses
import argparse

parser = argparse.ArgumentParser(description="Logistic Regression of K classes with D = 2.")
parser.add_argument("-t", action="store", dest="test", type=str, default='a',
                    help="t in [a, b]. Test a: Linear classification. Test b: Elliptic classification.")
parser.add_argument("-k", action="store", dest="numberOfClasses", type=int, default=2,
                    help="Number of classses used in test a.")

if __name__ == "__main__":
    results = parser.parse_args()
    if results.test == 'a':
        if results.numberOfClasses > 1:
            numberOfDataPerClass = np.random.uniform(80, 100, results.numberOfClasses)
            svg = ClassificationValuesGenerator(0, 30)

            classes, means = svg.getSyntheticValuesForClassification(numberOfDataPerClass, [[1, 0], [0, 1]])

            classificator = MCLogisticRegression(lambda x: [1, x[0], x[1]]) # linear
            classificator.findW(classes)

            classificable, means = svg.getSyntheticValuesForClassificationWithMeans([50] * results.numberOfClasses, [[1, 0], [0, 1]], means)
            
            classificated = [[] for i in range(0, results.numberOfClasses)]
            
            # using the same trng points.
            # classificable = classes
            
            for i in xrange(results.numberOfClasses):
                for point in classificable[i]:
                    cl = classificator.classificate(point)
                    classificated[cl].append(point)
                    print("point {0} in class {1} must be {2}".format(point, cl, i))
            
            plotClasses(classes, classificated, "classification")
        else:
            raise ValueError("Number of classes must be greater than 1")
    elif results.test == 'b':
        svg = ClassificationValuesGenerator(0, 30)

        classes, means = svg.getEllipticValuesForClassification()

        classificator = MCLogisticRegression(lambda x: [1, x[0], x[1], x[0]**2, x[0]*x[1], x[1]**2]) # elliptic # Works fine using LogisticRegression
        classificator.findW(classes)

        #TODO: Generate data for test

        classificated = [[] for i in range(0, 2)]
        
        # using the same trng points.
        classificable = classes
        
        for i in xrange(2):
            for point in classificable[i]:
                cl = classificator.classificate(point)
                classificated[cl].append(point)
                print("point {0} in class {1} must be {2}".format(point, cl, i))

        plotClasses(classes, classificated, "classification")
    else:
        raise ValueError("Test must be a or b.")