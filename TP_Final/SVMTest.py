#!/usr/bin/python
import numpy as np
import math
from SyntheticValues import ClassificationValuesGenerator
from Algorithms.SVM import SVM
from Plotter import plotClassesWithDecisionBoundary, plotCosts
import argparse
from random import shuffle

parser = argparse.ArgumentParser(description="Support Vector Machines for classification of 2 classes with vector space R^2.")
parser.add_argument("-t", action="store", dest="threshold", type=float, default=0.001,
                    help="Threshold for the loss function of the gradient descent.")
parser.add_argument("-lr", action="store", dest="learningRate", type=float, default=0.5,
                    help="Learning rate of the learning rate of the gradient descent.")
parser.add_argument("-c", action="store", dest="regParamC", type=float, default=0.1,
                    help="Regularization parameter of the slack variables: too small == hard margin | too large == soft margin.")
parser.add_argument("-i", action="store", dest="maxNumIter", type=int, default=10000,
                    help="Max number of iterations for the loss function of the gradient descent.")
parser.add_argument("-e", action="store", dest="testUsingTrainingData", type=int, default=1,
                    help="1: Test de classifier using a different data set. 0: Test using the training data set.")

def dataSetTestATraining():
    numberOfDataPerClass = np.random.uniform(80, 100, 2)
    svg = ClassificationValuesGenerator(0, 30)
    return svg.getSyntheticValuesForClassification(numberOfDataPerClass)
    
def dataSetTestATest(cov, means):
    svg = ClassificationValuesGenerator(0, 30)
    return svg.getSyntheticValuesForClassificationWithMeans([50] * 2, cov, means)

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
    '''
    X = [[5 ,8],[9 ,7],[2 ,8],[9 ,2],[2 ,5],[9 ,9],[9 ,9],[6 ,5],[9 ,2],[4 ,7],[6 ,4],[9 ,7],[2 ,8],[8 ,7],[2 ,3],[10,3],[10,9],[6 ,9],[9 ,5],[6 ,1],[5 ,10],[3 ,7],[5 ,8],[10,1],[5 ,3],[5 ,4],[1 ,5],[3 ,1],[2 ,9],[1 ,8],[3 ,5],[2 ,10],[6 ,8],[4 ,2],[5 ,1],[7 ,4],[8 ,6],[8 ,5],[9 ,1],[5 ,6],[1 ,5],[6 ,5],[7 ,5],[6 ,7],[9 ,4],[8 ,2],[7 ,9],[1 ,7],[1 ,4],[3 ,3]]
    Y = [1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1,1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1]
    svm = SVM()
    svm.train(X, Y)
    print('W:', svm.W)
    print('b:', svm.b)
    print('alphas:', svm.alphas)
    for p in zip(X, Y):
        c = svm.classificate(p[0])
        print('point {0} in class {1} must be {2}'.format(p[0], c, p[1]))
    '''    
    results = parser.parse_args()
    values = dataSetTestATraining()
    trainingData = values[0]
    cov = values[1]
    means = values[2]
    X1 = trainingData[0]
    Y1 = np.ones((len(X1),))
    X2 = trainingData[1]
    Y2 = -1*np.ones((len(X2),))
    X = np.concatenate((X1, X2), axis = 0)
    Y = np.append(Y1, Y2)
    classificator = SVM(results.threshold, results.learningRate, results.regParamC, results.maxNumIter)
    classificator.train(X, Y)
    print('W:', classificator.W)
    print('b:', classificator.b)
    if results.testUsingTrainingData == 0:
        classificateData(classificator, trainingData, trainingData, "supportVectorMachineTest")
    else:
        classificateData(classificator, trainingData, dataSetTestATest(cov, means), "supportVectorMachineTest")