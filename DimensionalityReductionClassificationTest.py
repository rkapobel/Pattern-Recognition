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
                    help="t in [a, b]. Test a: Fisher classification with 2 classes from 2-Dimensional space. Test b: Fisher classification with k classes from 3-Dimensional space. Default = a.")
parser.add_argument("-k", action="store", dest="numberOfClasses", type=int, default=2,
                    help="Number of classes used in test b. Default = 2.")
parser.add_argument("-e", action="store", dest="testUsingNewData", type=bool, default=True,
                    help="Test de classifier using a different data set.")

def dataSetTestATraining():
    numberOfDataPerClass = np.random.uniform(80, 100, 2)
    svg = ClassificationValuesGenerator(0, 10)
    return svg.getSyntheticValuesForClassification(numberOfDataPerClass, [[1, 0], [0, 1]])

def dataSetTestATest(means):
    svg = ClassificationValuesGenerator(0, 10)
    testData, means = svg.getSyntheticValuesForClassificationWithMeans([50] * results.numberOfClasses, [[1, 0], [0, 1]], means)
    return testData

def dataSetTestBTraining():
    numberOfDataPerClass = np.random.uniform(80, 100, results.numberOfClasses)
    svg = ClassificationValuesGenerator(0, 10)
    return svg.getSyntheticValuesForClassification(numberOfDataPerClass, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 3)
            
def dataSetTestBTest(means):
    svg = ClassificationValuesGenerator(0, 10)
    testData, means = svg.getSyntheticValuesForClassificationWithMeans([50] * results.numberOfClasses, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], means)
    return testData

def classificateData(classificator, trainingData, testData, numberOfClasses, fileName):
    classificated = [[] for i in range(0, numberOfClasses)]
        
    for i in xrange(numberOfClasses):
        for point in testData[i]:
            ci = classificator.classificate(point)
            classificated[ci].append(point)
            print("point {0} in class {1} must be {2}".format(point, ci, i))
    
    plotClasses(trainingData, classificated, fileName)

if __name__ == "__main__":
    results = parser.parse_args()
    if results.test == 'a':
        trainingData, means = dataSetTestATraining()

        lda = Fisher()
        lda.findW(trainingData[0], trainingData[1])

        if not results.testUsingNewData:
            classificateData(lda, trainingData, trainingData, 2, "linearDiscriminantAnalysisTestA")
        else:
            classificateData(lda, trainingData, dataSetTestATest(means), 2, "linearDiscriminantAnalysisTestA")

    elif results.test == 'b':
        if results.numberOfClasses > 1:
            trainingData, means = dataSetTestBTraining()
            
            lda = MCFisher()
            lda.findW(trainingData)

            trainingData = lda.reduceDimensionToClasses(trainingData)
            print('training data: {0}'.format(trainingData))
            classificator = MCLogisticRegression(lambda x: [1, x[0], x[1]]) # linear
            classificator.findW(trainingData)                

            if not results.testUsingNewData:
                classificateData(classificator, trainingData, trainingData, results.numberOfClasses, "linearDiscriminantAnalysisTestB")
            else:
                classificateData(classificator, trainingData, lda.reduceDimensionToClasses(dataSetTestBTest(means)), results.numberOfClasses, "linearDiscriminantAnalysisTestB")
        else:
            raise ValueError("Number of classes must be greater than 1")
    else:
        raise ValueError("Test must be a or b.")