#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 19:28:34 2017

@author: SpaceMeerkat
"""

import numpy as np
import matplotlib.pyplot as plt

#sigmoid function
def non_linear(x,deriv=False):
        if deriv == True:
                return x*(1-x)
        return 1./(1+np.exp(-x))

X = np.array([ [0,1,1],[1,1,1],[1,0,1],[0,1,0]])

y = np.array([[1,1,0,1]]).T

np.random.seed(1)

weights1 = np.random.uniform(-1,1,size=(3,4))
weights2 = np.random.uniform(-1,1,size=(4,1))
weights3 = np.random.uniform(-1,1,size=(4,1))

err = []

N = 50000

for j in range(N):
        
        l0 = X
        
        l1 = non_linear(np.dot(l0,weights1))
        l2 = non_linear(np.dot(l1,weights2))
        l3 = non_linear(np.dot(l2,weights3))
                
        l3_err = y - l3
              
        #error weighted derivative        
        l3_delta = l3_err * non_linear(l3,True)
        
        l2_err = l3_delta.dot(weights3.T)        
        
        l2_delta = l2_err * non_linear(l2,True)
        
        l1_err = l2_delta.dot(weights2.T)
        
        l1_delta = l1_err * non_linear(l1,True)
        
        weights3 += l2.T.dot(l3_delta)
        weights2 += l1.T.dot(l2_delta)
        weights1 += l0.T.dot(l1_delta)
