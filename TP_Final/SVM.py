#!/usr/bin/python
import numpy as np
from numpy.linalg import inv
import math
 
class Kernel:

	def __init__(self, phi, func):
		self.phi = phi
		self.func = func		
		
	def calculate(xi, xj):
		return func(phi(xi), phi(xj))

class SVM:

    def __init__(self, maxIter = 500, C = 1):
        self.maxIter = maxIter
		self.C = C
		self.W = None    
		self.b = None
		self.alphas = None
		self.X = None
		self.Y = None

	def maximizeDualProblem(X, Y):
		self.Y = np.array(Y)
		self.X = np.array(X) #TODO: Apply the kernel first to X
		D = self.X.shape
		self.alphas = self.getRandomAlphas(D[0])
		P = np.array([x*self.alphas*self.Y for x in self.X.T]).T
		A = np.dot(P, P.T)
		Max = sum(self.alphas) - 0.5*sum(sum(A))
		MaxAux = Max
		it = 0
		R1 = self.alphaValidation(self.alphas) #TODO: Figure out the correct way to find the alphas
		while (not R1) and it < self.maxIter:
			alphasAux = self.getRandomAlphas(D[0])
			P = np.array([x*alphasAux*self.Y for x in self.X.T]).T
			A = np.dot(P, P.T)
			MaxAux = sum(alphasAux) - 0.5*sum(sum(A))
			R1 = self.alphaValidation(self.alphas)
			if MaxAux > Max and R1:
				Max = MaxAux
				self.alphas = alphasAux
			it += 1
		self.W = sum(np.dot(self.X.T, self.alphas*self.Y))
		self.findB()

		def getRandomAlphas(l):
			return numpy.random.uniform(low=0, high=self.C, size=(l,1))

		def alphaValidation(alphas):
			np.dot(alphas, self.Y) == 0
	
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
			return np.sign(np.dot(np.dot(X, x), self.alphas*self.Y) + self.b) #TODO: Apply the kernel to x
