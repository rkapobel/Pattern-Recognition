#!/usr/bin/python
import numpy as np
import math
from SyntheticValues import ClassificationValuesGenerator
from Algorithms.SVM import SVM
from Plotter import plotClassesWithDecisionBoundary, plotConvergence
import argparse
from random import shuffle

parser = argparse.ArgumentParser(description="Support Vector Machines for classification of 2 classes with vector space R^2.")
parser.add_argument("-lr", action="store", dest="learningRate", type=float, default=0.1,
                    help="Learning rate of the learning rate of the gradient descent.")
parser.add_argument("-c_list", action="store", dest="regListParamC", nargs="+", default=0.25, type=float,
                    help="A list of regularization parameter for the slack variables: too small == hard margin | too large == soft margin.") 
parser.add_argument("-i", action="store", dest="maxNumIter", type=int, default=100,
                    help="Max number of iterations for the loss function of the gradient descent.")
parser.add_argument("-e", action="store", dest="testUsingTrainingData", type=int, default=1,
                    help="1: Test the classifier using a different data set. 0: Test using the training data set.")

def dataSetTestATraining():
    numberOfDataPerClass = np.random.uniform(80, 100, 2)
    svg = ClassificationValuesGenerator(0, 30)
    return svg.getSyntheticValuesForClassification(numberOfDataPerClass)
    
def dataSetTestATest(cov, means):
    svg = ClassificationValuesGenerator(0, 1)
    return svg.getSyntheticValuesForClassificationWithMeans([50] * 2, cov, means)

def dataSetTestATrainingWithFixedDistribution():
    numberOfDataPerClass = np.random.uniform(80, 100, 2)
    svg = ClassificationValuesGenerator()
    cov = np.array([[1, 0], [0, 1]])
    means = [[0, 0], [0, 8]]
    return [svg.getSyntheticValuesForClassificationWithMeans(numberOfDataPerClass, cov, means), cov, means]

def classificateData(classificator, trainingData, testData, fileName):
    classificated = [[] for i in range(0, 2)]
    for i in xrange(2):
        for point in testData[i]:
            ci = classificator.classificate(point)
            if ci == 1:
                ci = 0
            else: 
                ci = 1
            classificated[ci].append(point)
            print("point {0} in class {1} must be {2}".format(point, ci, i))
    
    plotClassesWithDecisionBoundary(trainingData, classificated, classificator.W, classificator.b, fileName)
    #plotConvergence(classificator.getEpochs(), classificator.costs, fileName + 'costFunction', 'Cost function (Loss)')

if __name__ == "__main__":
    results = parser.parse_args()
    # To test with fixed distribution. set them in the method.
    # TODO: Automatize this.
    #trainingValues = dataSetTestATrainingWithFixedDistribution()
    # To test with random distribution. 
    trainingValues = dataSetTestATraining()
    trainingData = trainingValues[0]
    cov = trainingValues[1]
    print(cov, 'cov')
    means = trainingValues[2]
    print(means, 'means')
    testValues = dataSetTestATest(cov, means)
    X1 = trainingData[0]
    Y1 = np.ones((len(X1),))
    X2 = trainingData[1]
    Y2 = -1*np.ones((len(X2),))
    X = np.concatenate((X1, X2), axis = 0)
    Y = np.append(Y1, Y2)
    for C in results.regListParamC:
        classificator = SVM(results.learningRate, C, results.maxNumIter)
        classificator.train(X, Y)
        print('W:', classificator.W)
        print('b:', classificator.b)
        if results.testUsingTrainingData == 0:
            classificateData(classificator, 
            trainingData, 
            trainingData, 
            "supportVectorMachineTest-lr:" + 
            str(results.learningRate) + 
            "-C:" + str(C) + 
            "-i:" + str(results.maxNumIter) +
            "-means:" + str(means) +
            "-cov:" + str(cov))
        else:
            classificateData(
                classificator, 
                trainingData, 
                testValues, 
                "supportVectorMachineTest-lr:" + 
            str(results.learningRate) + 
            "-C:" + str(C) + 
            "-i:" + str(results.maxNumIter) +
            "-means:" + str(means) +
            "-cov:" + str(cov))