#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 14:00:27 2018

@author: jamesdawson
"""
'''
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

        def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 6, 5)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
                x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
                x = F.max_pool2d(F.relu(self.conv2(x)), 2)
                x = x.view(-1, self.num_flat_features(x))
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        
        def num_flat_features(self, x):
                size = x.size()[1:]  
                num_features = 1
                for s in size:
                    num_features *= s
                return num_features

net = Net()
print(net)


params = list(net.parameters())
print(len(params))
print(params[0].size()) 


input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)
print(out)


net.zero_grad()
out.backward(torch.randn(1, 10))

'''
'''
import torch
from torch import np
from torch import nn

X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]]).astype('float32')
Y = np.array([[0],[1],[1],[0]]).astype('float32')

net = nn.Sequential()
net.add(nn.Linear(3, 4))
net.add(nn.Sigmoid())
net.add(nn.Linear(4, 1))
net.add(nn.Sigmoid())
net.float()

crit = nn.MSELoss()
crit.float()

for j in range(2400):
        net.zero_grad()
        output = net.forward(X)
        loss = crit.forward(output, Y)
        gradOutput= crit.backward(output, Y)
        gradInput = net.backward(X, gradOutput)
        net.updateParameters(1)
        if (j%200) == 0:
                print("Error:" + str(loss))


'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F	
import torch.optim as optim
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 50) # 2 Input noses, 50 in middle layers
        self.fc2 = nn.Linear(50, 1) # 50 middle layer, 1 output nodes
	    self.rl1 = nn.ReLU()
	    self.rl2 = nn.ReLU()
	
    def forward(self, x):
        x = self.fc1(x)
	x = self.rl1(x)
	x = self.fc2(x)
	x = self.rl2(x)
        return x

if __name__ == "__main__":
	## Create Network

	net = Net()
	#print net

	## Optimization and Loss

	#criterion = nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss
	criterion = nn.MSELoss()
	#criterion = nn.L1Loss()
	#criterion = nn.NLLLoss()
	#criterion = nn.BCELoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.1)
	#optimizer = optim.Adam(net.parameters(), lr=0.01)

	trainingdataX = [[[0.01, 0.01], [0.01, 0.90], [0.90, 0.01], [0.95, 0.95]], [[0.02, 0.03], [0.04, 0.95], [0.97, 0.02], [0.96, 0.95]]]
	trainingdataY = [[[0.01], [0.90], [0.90], [0.01]], [[0.04], [0.97], [0.98], [0.1]]]
 	NumEpoches = 20000
	for epoch in range(NumEpoches):

		running_loss = 0.0
		for i, data in enumerate(trainingdataX, 0):
			inputs = data
			labels = trainingdataY[i]
			inputs = Variable(torch.FloatTensor(inputs))
			labels = Variable(torch.FloatTensor(labels))
			optimizer.zero_grad()
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()        
			optimizer.step()
			running_loss += loss.data[0]
			if i % 1000 == 0:
				print "loss: ", running_loss
				running_loss = 0.0
	print "Finished training..."
	print net(Variable(torch.FloatTensor(trainingdataX[0])))
		
	
	


		







