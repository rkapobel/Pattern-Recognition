#!/usr/bin/python
import numpy as np
import math
from SyntheticValues import ClassificationValuesGenerator
from Algorithms.SVM import SVM
from Plotter import plotClassesWithDecisionBoundary, plotCosts
import argparse
from random import shuffle

parser = argparse.ArgumentParser(description="Support Vector Machines for classification of 2 classes with vector space R^2.")
parser.add_argument("-lr", action="store", dest="learningRate", type=float, default=0.5,
                    help="Learning rate of the learning rate of the gradient descent.")
parser.add_argument("-c_list", action="store", dest="regListParamC", nargs="+", default=0.1, type=float,
                    help="A list of regularization parameter for the slack variables: too small == hard margin | too large == soft margin.") 
parser.add_argument("-i", action="store", dest="maxNumIter", type=int, default=20000,
                    help="Max number of iterations for the loss function of the gradient descent.")
parser.add_argument("-e", action="store", dest="testUsingTrainingData", type=int, default=1,
                    help="1: Test de classifier using a different data set. 0: Test using the training data set.")

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
    means = [[0, 0], [0, 5]]
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
    #TODO: I need costs for every class with the actual implementation of the plotter.
    #plotCosts(classificator.getEpochs(), classificator.costs, fileName + 'costFunction')

if __name__ == "__main__":
    results = parser.parse_args()
    values = dataSetTestATrainingWithFixedDistribution()
    #TODO: to uncomment after testing
    #values = dataSetTestATraining()
    trainingData = values[0]
    cov = values[1]
    print(cov, 'cov')
    means = values[2]
    print(means, 'means')
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
            classificateData(classificator, trainingData, trainingData, "supportVectorMachineTest+C:" + str(C))
        else:
            classificateData(classificator, trainingData, dataSetTestATest(cov, means), "supportVectorMachineTest+C:" + str(C))