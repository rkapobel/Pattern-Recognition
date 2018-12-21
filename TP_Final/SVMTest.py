#!/usr/bin/python
import numpy as np
import math
#from SyntheticValues import ClassificationValuesGenerator
from SVM import SVM
#from Plotter import plotClasses, plotConvergence
import argparse
from random import shuffle

if __name__ == "__main__":
    X = [[5 ,8],[9 ,7],[2 ,8],[9 ,2],[2 ,5],[9 ,9],[9 ,9],[6 ,5],[9 ,2],[4 ,7],[6 ,4],[9 ,7],[2 ,8],[8 ,7],[2 ,3],[10,3],[10,9],[6 ,9],[9 ,5],[6 ,1],[5 ,10],[3 ,7],[5 ,8],[10,1],[5 ,3],[5 ,4],[1 ,5],[3 ,1],[2 ,9],[1 ,8],[3 ,5],[2 ,10],[6 ,8],[4 ,2],[5 ,1],[7 ,4],[8 ,6],[8 ,5],[9 ,1],[5 ,6],[1 ,5],[6 ,5],[7 ,5],[6 ,7],[9 ,4],[8 ,2],[7 ,9],[1 ,7],[1 ,4],[3 ,3]]
    Y = [1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1,1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1]
    svm = SVM()
    svm.maximizeDualProblem(X, Y)
    print(svm.W, 'W')
    print(svm.b, 'b')
    print(svm.alphas, 'alphas')
    print(svm.classificate([5,8]), 'x is:')
    print(svm.classificate([2,8]), 'x is:')
    print(svm.classificate([9,2]), 'x is:')
    print(svm.classificate([2,4]), 'x is:')
    print(svm.classificate([9,9]), 'x is:')