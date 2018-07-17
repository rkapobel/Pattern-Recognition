#!/usr/bin/python
import numpy as np
import math
from SyntheticValues import ClassificationValuesGenerator, getEllipticValuesForClassification
from Algorithms.LogisticRegression import NRLogisticRegression, MCLogisticRegression
from Algorithms.LogisticRegression import LINEAR, CIRCULAR, ELLIPTIC
from Plotter import plotClasses, plotCosts
import argparse

parser = argparse.ArgumentParser(description="Logistic Regression of K  trainingData with D = 2.")
parser.add_argument("-t", action="store", dest="test", type=str, default='a',
                    help="t in [a, b]. Test a: Linear classification. Test b: Elliptic classification.")
parser.add_argument("-k", action="store", dest="numberOfClasses", type=int, default=2,
                    help="Number of classses used in test a. Default = 2.")
parser.add_argument("-e", action="store", dest="testUsingNewData", type=bool, default=True,
                    help="Test de classifier using a different data set.")

def dataSetTestATraining():
    numberOfDataPerClass = np.random.uniform(80, 100, results.numberOfClasses)
    svg = ClassificationValuesGenerator(0, 30)
    return svg.getSyntheticValuesForClassification(numberOfDataPerClass, [[1, 0], [0, 1]])
    
def dataSetTestATest(means):
    svg = ClassificationValuesGenerator(0, 30)
    testData, means = svg.getSyntheticValuesForClassificationWithMeans([50] * results.numberOfClasses, [[1, 0], [0, 1]], means)
    return  testData

def classificateData(classificator, trainingData, testData, numberOfClasses, fileName):
    classificated = [[] for i in range(0, numberOfClasses)]
        
    for i in xrange(numberOfClasses):
        for point in testData[i]:
            ci = classificator.classificate(point)
            classificated[ci].append(point)
            print("point {0} in class {1} must be {2}".format(point, ci, i))
    
    plotClasses(trainingData, classificated, fileName)
    plotCosts(classificator.epochs, classificator.costs, fileName + 'costFunction')

if __name__ == "__main__":
    results = parser.parse_args()
    if results.test == 'a':
        if results.numberOfClasses > 1:
            trainingData, means = dataSetTestATraining()
            classificator = MCLogisticRegression(LINEAR)
            classificator.findW(trainingData)
                       
            if not results.testUsingNewData:
                classificateData(classificator, trainingData, trainingData, results.numberOfClasses, "logisticRegressionTestA")
            else:
                classificateData(classificator, trainingData, dataSetTestATest(means), results.numberOfClasses, "logisticRegressionTestA")
        else:
            raise ValueError("Number of trainingData must be greater than 1")
    elif results.test == 'b':
        trainingData, means = getEllipticValuesForClassification()
        classificator = NRLogisticRegression(ELLIPTIC)
        #classificator = MCLogisticRegression(ELLIPTIC, regularize=True)
        classificator.findW(trainingData)

        if not results.testUsingNewData:
            classificateData(classificator, trainingData,  trainingData, 2, "logisticRegressionTestB")
        else:
            classificateData(classificator, trainingData, getEllipticValuesForClassification()[0], 2, "logisticRegressionTestB")
    else:
        raise ValueError("Test must be a or b.")