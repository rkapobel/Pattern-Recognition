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
		
	def maximizeDualProblem(X, Y):
		Y = np.array(Y)
		X = np.array(X) #TODO: Apply the kernel
		D = X.shape
		self.alphas = self.getRandomAlphas(D[0])
		P = np.array([x*self.alphas*Y for x in X.T]).T
		A = np.dot(P, P.T)
		Max = sum(self.alphas) - 0.5*sum(sum(A))
		MaxAux = Max
		it = 0
		R1 = self.r1(self.alphas, Y) #TODO: Figure out the correct way to find the alphas
		while (not R1) and it < self.maxIter:
			alphasAux = self.getRandomAlphas(D[0])
			P = np.array([x*alphasAux*Y for x in X.T]).T
			A = np.dot(P, P.T)
			MaxAux = sum(alphasAux) - 0.5*sum(sum(A))
			R1 = self.r1(self.alphas, Y)
			if MaxAux > Max and R1:
				Max = MaxAux
				self.alphas = alphasAux
			it += 1
		self.W = sum(np.dot(X.T, self.alphas*Y))
		#TODO: Find self.b

		def getRandomAlphas(l):
			return numpy.random.uniform(low=0, high=self.C, size=(l,1))

		def r1(alphas, Y):
			np.dot(alphas, Y) == 0

		#TODO: A function to classificate a point is needed