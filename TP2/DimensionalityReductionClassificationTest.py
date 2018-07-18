#!/usr/bin/python
import numpy as np
import math
from SyntheticValues import ClassificationValuesGenerator
from Algorithms.DimensionalityReduction import Fisher, MCFisher
from Algorithms.LogisticRegression import MCLogisticRegression
from Plotter import plotClasses
import argparse

parser = argparse.ArgumentParser(description="Fisher linear discriminant analysis.")
parser.add_argument("-t", action="store", dest="test", type=str, default='a',
                    help="t in [a, b]. Test a: Fisher classification with 2 classes from 2-Dimensional space. Test b: Fisher classification with 3 classes from 3-Dimensional space. Default = a.")
parser.add_argument("-e", action="store", dest="testUsingTrainingData", type=int, default=1,
                    help="1: Test de classifier using a different data set. 0: Test using the training data set.")
parser.add_argument("-d", action="store", dest="dimension", type=int, default=0,
                    help="If not indicated, test B will run with D = 3. Must be greater or equal than 3.")

def dataSetTestATraining(dim):
    numberOfDataPerClass = np.random.uniform(80, 100, 2)
    svg = ClassificationValuesGenerator(0, 10)
    return svg.getSyntheticValuesForClassification(numberOfDataPerClass, dim)

def dataSetTestATest(cov, means):
    svg = ClassificationValuesGenerator(0, 20)
    return svg.getSyntheticValuesForClassificationWithMeans([50] * 2, cov, means)
    
def dataSetTestBTraining(dim):
    numberOfDataPerClass = np.random.uniform(80, 100, 3)
    svg = ClassificationValuesGenerator(0, 10)
    return svg.getSyntheticValuesForClassification(numberOfDataPerClass, dim)
            
def dataSetTestBTest(cov, means):
    svg = ClassificationValuesGenerator(0, 20)
    return svg.getSyntheticValuesForClassificationWithMeans([50] * 3, cov, means)
    
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
        values = dataSetTestATraining(2)
        
        trainingData = values[0]
        cov = values[1]
        means = values[2]

        lda = Fisher()
        lda.findW(trainingData[0], trainingData[1])

        if results.testUsingTrainingData == 0:
            classificateData(lda, trainingData, trainingData, 2, "linearDiscriminantAnalysisTestA")
        else:
            classificateData(lda, trainingData, dataSetTestATest(cov, means), 2, "linearDiscriminantAnalysisTestA")

    elif results.test == 'b':
        dim = 3
        if results.dimension > 3:
            dim = results.dimension

        values = dataSetTestBTraining(dim)

        trainingData = values[0]
        cov = values[1]
        means = values[2]
        
        lda = MCFisher()
        lda.findW(trainingData)

        trainingData = lda.reduceDimensionToClasses(trainingData)
        classificator = MCLogisticRegression(lambda x: [1, x[0], x[1]]) # linear
        classificator.findW(trainingData)                
        
        if results.testUsingTrainingData == 0:
            classificateData(classificator, trainingData, trainingData, 3, "linearDiscriminantAnalysisTestB")
        else:
            classificateData(classificator, trainingData, lda.reduceDimensionToClasses(dataSetTestBTest(cov, means)), 3, "linearDiscriminantAnalysisTestB")
    else:
        raise ValueError("Test must be a or b.")