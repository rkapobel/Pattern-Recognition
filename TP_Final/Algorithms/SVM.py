#!/usr/bin/python
import numpy as np
from numpy.linalg import inv
import math

class SVM:

	def __init__(self, maxIter = 500, C = 1, lr = 0.5):
		self.maxIter = maxIter
		self.C = C
		self.lr = lr
		self.costs = []

	def train(self, X, Y):
		self.Y = np.array(Y)
		self.X = np.array(X)
		self.D = self.X.shape
		alphas = None
		alphasAux = self.getRandomAlphas(self.D[0])
		allCosts = []
		it = 0
		while it < self.maxIter and self.validAlphas(alphasAux):
			alphas = alphasAux
			alphasAux += self.lr*(self.calculateJ(alphasAux)*np.zeros((self.D[0],))) # This is gradient ascent to maximize alphas
			self.findW()
			self.findB()
			cost = self.calculateCostFunction()
            allCosts.append(cost)
			it += 1
		self.alphas = alphas
		self.findW()
		self.findB()
		self.costs.append(allCosts)
		
	def getRandomAlphas(self, l):
		return np.random.uniform(low=0, high=self.C, size=(l,))

	def validAlphas(self, alphas):
		for v in alphas:
			if v < 0 or v > self.C:
				return False
		return True

	def calculateJ(self, alphas):
		P = np.array([sum(x*alphas*self.Y) for x in self.X.T]).T
		A = np.outer(P, P)
		return sum(alphas) - 0.5*sum(sum(A))	

	def findW(self):
		self.W = sum([t[0]*t[1] for t in zip(self.X, self.alphas*self.Y)])

	def findB(self):
		for i in range(self.X.shape[0]):
			xi = self.X[i]
			yi = self.Y[i]
			prod = np.dot(self.W, xi)
			b = yi - prod
			if yi*(prod + b) == 1:
				self.b = b
				return
		#TODO: throw an error?

	def classificate(self, x):
		return np.sign(np.dot(np.dot(self.X, x), self.alphas*self.Y) + self.b)
	
	def calculateCostFunction(self):
        return (1 / float(self.D[0])) * sum([self.costFunction(p[0], p[1]) for p in zip(self.X, self.Y)])
        
    def costFunction(self, xi, yi):
        return max(0, 1 - yi*(np.dot(self.W, xi) - self.b))

    def getEpochs(self):
        return list(range(self.maxIter))
