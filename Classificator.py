#!/usr/bin/python
import numpy as np
from numpy.linalg import inv
import math

class Classificator:
    W = None

    def findW(self, classes):
        T = None
        X = None

        for i in len(classes):
            aClass = classes[i]
            t = [0] * len(classes)
            t[i] = 1
            [X.append([1, x1, x2]) for x1, x2 in zip(aClass)]
            T.extend([t] * len(aClass))
            
        Xt = X.T
        try:
            XtXInv = inv(np.dot(Xt, X))
            self.W = np.dot(np.dot(XtXInv, Xt), T)
        except np.linalg.LinAlgError as e:
            print(e)

    def classificate(self, x1, x2):
        Y = np.dot(self.W, [x1, x2])
        return np.argmax(Y)