#!/usr/bin/python
import numpy as np
from numpy.linalg import inv
from random import shuffle, choice
import math


class SVM:

	def __init__(self, lr = 0.1, C = 0.25, maxNumIter = 100):
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
		self.numIter = 0
		data = zip(self.X, self.Y)
		while self.numIter < self.maxNumIter:
			loss = 0
			shuffle(data)
			(xi, yi) = choice(data)
			prod = np.dot(self.W, xi)
			self.b = yi - prod
			gamma = (self.lr / float(1 + self.numIter))
			for (xi, yi) in data:
				v = yi * (np.dot(self.W, xi) + self.b)
				loss += max(0, 1 - v)
				Wgrad = -xi * yi if v < 1 else 0
				bgrad = -yi if v < 1 else 0
				self.W -= gamma * (self.W + self.C * Wgrad)
				self.b -= gamma * self.C * bgrad
			self.costs.append(loss)
			self.numIter += 1
		#print("costs:", self.costs)

	def classificate(self, x):
		return int(np.sign(np.dot(self.W, x) + self.b))

	def getEpochs(self):
		return list(range(self.numIter))
