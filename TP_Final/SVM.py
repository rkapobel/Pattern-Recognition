#!/usr/bin/python
import numpy as np
from numpy.linalg import inv
import math

class SVM:

    def __init__(self, maxIter = 500, C = 1):
        self.maxIter = maxIter
		self.C = C
		self.W = None    
		self.b = None
		self.alphas = None
		self.X = None
		self.Y = None
		self.D = None

	def maximizeDualProblem(X, Y):
		self.Y = np.array(Y)
		self.X = np.array(X)
		self.D = self.X.shape
		alphas = None
		alphasAux = self.getRandomAlphas(self.D[0])
		it = 0
		while it < self.maxIter and validAlphas(alphasAux):
			alphas = alphasAux
			alphasAux += self.calculateJ(alphasAux)*np.zeros((self.D[0], )) # This is gradient ascent to maximize alphas
			it += 1
		self.findW()
		self.findB()
		self.alphas = alphas

		def getRandomAlphas(l):
			return numpy.random.uniform(low=0, high=self.C, size=(l,1))

		def validAlphas(alphas):
			for v in alphas:
				if v < 0 or v > self.C:
					return False
			return True

		def calculateJ(alphas):
			P = np.array([x*alphas*self.Y for x in self.X.T]).T
			A = np.dot(P, P.T)
			return sum(alphas) - 0.5*sum(sum(A))	

		def findW():
			self.W = sum(np.dot(self.X.T, self.alphas*self.Y))

		def findB():
			for i in range(self.X.shape[0]):
				xi = self.X[i]
				yi = self.Y[i]
				prod = np.dot(self.W, xi)
				b = yi - prod
				if yi*(prod + b) == 1:
					self.b = b
					return
			#TODO: throw an error?

		def classificate(x):
			return np.sign(np.dot(np.dot(X, x), self.alphas*self.Y) + self.b)
