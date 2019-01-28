#!/usr/bin/python
import numpy as np
from numpy.linalg import inv
from random import shuffle
import math


class SVM:

	def __init__(self, lr = 0.5, C = 0.1, maxNumIter = 10000):
		self.lr = lr
		self.C = C
		self.costs = []
		self.numIter = 0
		self.maxNumIter = maxNumIter

	def train(self, X, Y):
		self.Y = np.array(Y)
		self.X = np.array(X)
		self.D = self.X.shape
		self.W = np.zeros(self.D[1])
		self.b = 0
		self.costs = []
		allCosts = []
		loss = 0
		lossAux = np.inf
		self.numIter = 0
		delta = np.abs(lossAux - loss)
		while self.numIter < self.maxNumIter:
			sumST = 0 # It's a vector indeed, but np allows vector + numbers and other operations
			loss = 0
			data = zip(self.X, self.Y)
			shuffle(data)
			gamma = (self.lr / float(self.numIter + 1))
			for (xi, yi) in data:
				v = yi * (np.dot(self.W, xi) + self.b)
				loss += max(0, 1 - v)
				#sumST += -xi * yi if (yi * np.dot(self.W, xi)) < 1 else 0
				grad = -xi * yi if (yi * np.dot(self.W, xi)) < 1 else 0
				self.W -= gamma * (self.W + self.C * grad)
			#self.W -= gamma * self.C * (sumST / np.linalg.norm(sumST))
			self.findB()
			loss *= self.C
			delta = np.abs(lossAux - loss)
			allCosts.append(delta)
			lossAux = loss
			self.numIter += 1
		self.costs.append(allCosts) #TODO: It is the cost == delta of loss_{i} - loss_{i-1} ?
		#print("costs:", self.costs)

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
		return int(np.sign(np.dot(self.W, x) + self.b))

	def getEpochs(self):
		return list(range(self.numIter))
