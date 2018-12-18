#!/usr/bin/python
import numpy as np
from numpy.linalg import inv
import math
 
class Kernel:

	def __init__(self, phi, func):
		self.phi = phi
		self.func = func		
		pass

	def calculate(xi, xj):
		return func(phi(xi), phi(xj))

class SVM:

    def __init__(self, maxIter = 200, C = 1):
        self.maxIter = maxIter
		self.C = C
		self.W = None    
		self.b = None
		self.reg = None
		
