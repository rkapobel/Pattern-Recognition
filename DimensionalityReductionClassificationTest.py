#!/usr/bin/python
import numpy as np
import math
from SyntheticValues import ClassificationValuesGenerator
from DimensionalityReduction import Fisher, MCFisher
from LogisticRegression import MCLogisticRegression
from Plotter import plotClasses
import argparse

parser = argparse.ArgumentParser(description="Fisher linear discriminant analysis.")
parser.add_argument("-t", action="store", dest="test", type=str, default='a',
                    help="t in [a, b]. Test a: Fisher classification with 2 classes from 2-Dimensional space. Test b: Fisher classification with k classes from 3-Dimensional space. Default = a")
parser.add_argument("-k", action="store", dest="numberOfClasses", type=int, default=2,
                    help="k >= 2 only valid in test b. Default = 2")

if __name__ == "__main__":
    results = parser.parse_args()
    if results.test == 'a':
        numberOfDataPerClass = np.random.uniform(80, 100, 2)
        svg = ClassificationValuesGenerator(0, 10)
        classes, means = svg.getSyntheticValuesForClassification(numberOfDataPerClass, [[1, 0], [0, 1]], 2)

        lda = Fisher()
        lda.findW(classes[0], classes[1])

        classificable, means = svg.getSyntheticValuesForClassificationWithMeans([50] * results.numberOfClasses, [[1, 0], [0, 1]], means)
        classificated = [[] for i in range(0, 2)]
        
        # using the same trng points.
        # classificable = classes
        
        for i in xrange(2):
            for point in classificable[i]:
                cl = lda.classificate(point)
                classificated[cl].append(point)
                print("point {0} in class {1} must be {2}".format(point, cl, i))
        
        plotClasses(classes, classificated, "classification")
    elif results.test == 'b':
        if results.numberOfClasses > 1:
            numberOfDataPerClass = np.random.uniform(80, 100, results.numberOfClasses)
            svg = ClassificationValuesGenerator(0, 10)
            classes, means = svg.getSyntheticValuesForClassification(numberOfDataPerClass, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 3)
            
            lda = MCFisher()
            lda.findW(classes)

            classificator = MCLogisticRegression(lambda x: x) # identity
            classificator.findW(lda.reduceDimensionToClasses(classes))                

            classificable, means = svg.getSyntheticValuesForClassificationWithMeans([50] * results.numberOfClasses, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], means)
            classificated = [[] for i in range(0, results.numberOfClasses)]

            # using the same trng points.
            # classificable = classes
            
            for i in xrange(results.numberOfClasses):
                for point in classificable[i]:
                    cl = classificator.classificate(lda.reduceDimension(point))
                    classificated[cl].append(point)
                    print("point {0} in class {1} must be {2}".format(point, cl, i))
            
            plotClasses(classes, classificated, "classification")
        else:
            raise ValueError("Number of classes must be greater than 1")
    else:
        raise ValueError("Test must be a or b.")