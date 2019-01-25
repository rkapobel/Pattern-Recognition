#!/usr/bin/python
import numpy as np
from numpy.linalg import inv
from random import shuffle
import math


class SVM:

	def __init__(self, thresh = 0.001, lr = 0.5, C = 0.1, maxNumIter = 10000):
		self.thresh = thresh
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
		self.numIter = 1
		delta = np.abs(lossAux - loss)
		while self.numIter < self.maxNumIter:
			sumST = 0
			loss = 0
			data = zip(self.X, self.Y)
			shuffle(data)
			for (xi, yi) in data:
				v = yi*(np.dot(self.W, xi) + self.b)
				loss += max(0, 1 - v)
				#sumST += -xi*yi if v < 1 else 0
				self.W -= (self.lr / float(self.numIter)) * (self.W + self.C*(-xi*yi if v < 1 else 0))
			loss *= 1/float(self.D[0]) #TODO: or self.C*loss ?
			#self.W -= (self.lr / float(self.numIter)) * (self.W + self.C*sumST) #TODO: (1/float(self.D[0])) or self.C ?
			self.findB()
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