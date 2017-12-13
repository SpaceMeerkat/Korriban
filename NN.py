#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 19:28:34 2017

@author: jamesdawson
"""

import numpy as np
import matplotlib.pyplot as plt

#sigmoid function
def non_linear(x,deriv=False):
        if deriv == True:
                return x*(1-x)
        return 1./(1+np.exp(-x))

def Neural_Network(X,y,iterations,layers):
        
        weights_init = np.random.uniform(-1,1,size=(len(X[0]),len(X)))
        weights_final =  np.random.uniform(-1,1,size=(len(X),1))
        
        l_inter = np.zeros([layers,len(X),len(X)])
        weights_inter = np.random.uniform(-1,1,size=(l_inter.shape))
        l_inter_err = np.zeros([layers,len(X),len(X)])
        l_inter_delta = np.zeros([layers,len(X),len(X)])
        
        for j in range(iterations):
                
                l_init = X    
                
                   
                l_inter[0] = non_linear(np.dot(l_init,weights_init))        
                for i in range(1,layers):                
                        l_inter[i] = non_linear(np.dot(l_inter[i-1],weights_inter[i-1]))
                l_final = non_linear(np.dot(l_inter[-1],weights_final))
                
                
                l_final_err = y - l_final
                l_final_delta = l_final_err * non_linear(l_final,True)
                
                
                l_inter_err[-1] = l_final_delta.dot(weights_final.T)
                l_inter_delta[-1] = l_inter_err[-1] * non_linear(l_inter[-1],True)          
                for k in range(2,layers+1):         
                        l_inter_err[-k] = l_inter_delta[-(k-1)].dot(weights_inter[-(k-1)].T)
                        l_inter_delta[-k] = l_inter_err[-k] * non_linear(l_inter[-k],True)                                 
                weights_final += l_inter[-1].T.dot(l_final_delta)
                
                for l in range(1,layers):
                        weights_inter += l_inter[-l].T.dot(l_inter_delta[-l])
                        
                weights_init += l_init.T.dot(l_inter_delta[0])
        
        return l_final

###############################################################################
layers = 3   
iterations = 100000
X = np.array([ [0,1,1],[1,1,1],[1,0,1],[0,0,0]])
y = np.array([[0,1,1,0]]).T
results = Neural_Network(X,y,iterations,layers)
print(results)

