#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:41:47 2019

@author: sajedrakhshani
"""


import numpy as np
from matplotlib import pyplot as plt, cm
from sklearn import datasets
from sklearn.decomposition import pca
from sklearn.metrics import confusion_matrix

plt.close()

#%% prepare data:
y = np.zeros((100,2))
x = np.zeros((100, 2))
x[0:25,:] = np.random.randn(25,2) + np.array([[-3,-3]])

x[25:50, :] = np.random.randn(25,2) + np.array([[3,-3]])
y[25:50, 0] = 0
y[25:50, 1] = 1

x[50:75, :] = np.random.randn(25,2) + np.array([[-3,3]])
y[50:75, 0] = 1
y[50:75, 1] = 0

x[75:100, :] = np.random.randn(25,2) + np.array([[3,3]])
y[75:100, 0] = 1
y[75:100, 1] = 1

x = x / abs(x).max()


#%% first plot
#plt.scatter(x[:,0], x[:,1], c=['red']*25+['orange']*25+['green']*25+['blue']*25)

#%% some function to use
def hardlim(x):
    return 1.0*(x >= 0)

def plot_hyperplane(w, b, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx - (b) / w[1]
    plt.plot(xx, yy, linestyle, label=label)
    

def bin2dec(x):
    m, n = x.shape
    y = np.zeros((m,1))
    for i in range(m):
        t = x[i,:]
        y[i] = sum([k*(2**(n-1-j)) for j,k in enumerate(t)])
    return y


def predictovermesh(W, b, xx, yy):
    D = np.vstack((xx.ravel(), yy.ravel())).T
    Z = (hardlim(W.dot(D.T) + b)).T
    #Z = ((W.dot(D.T) + b)).T
    #Z1 = Z[:,0]
    #Z2 = Z[:,1]
    Z = bin2dec(Z)

    Z = Z.reshape(xx.shape)
    #Z1 = Z1.reshape(xx.shape)
    #Z2 = Z2.reshape(xx.shape)
    #Z = Z1 * Z2
    #Z = sigmoid(Z)
    #Z = Z[:,1]#+Z[:,1]
    #Z0 = Z[:,0].reshape(xx.shape)
    #Z1 = Z[:,1].reshape(xx.shape)
    #Z = Z0*Z1
    return Z


def sigmoid(z):
    return 1 / (1 + np.exp(z))
#%% initiate the network:
W = np.random.rand(2,2)*1;
b = np.random.rand(2,1)*1;
#%% train W and b on "trainX" and "trainY" data
#------- write your code here:---------#
N_sample = 100
P_matrix = x[0:N_sample,:]
T_matrix = y[0:N_sample,:]
index = 0
max_iter = 15000
iters = 0
criterion_check = 0;
while criterion_check == 0:
    sample = P_matrix[index,::].reshape(-1,1)  # shape is 2x1
    target = T_matrix[index,::].reshape(-1,1)  # shape is 2x1
    
    output = hardlim(W.dot(sample) + b)
    error = target - output
    
    W = W + error.dot(sample.T)
    b = b + error
    index += 1
    index = index % N_sample
    
    all_error = (hardlim(W.dot(P_matrix.T) + b)).T - T_matrix
    
    if np.all(np.all(np.nonzero(all_error))) == True or iters > max_iter:
        criterion_check = 1
    iters += 1

print('training have been done in {} iterations'.format([iters]))
#%%  predict output:
pr = np.zeros((100,2));
for i in range(100):
    pr[i]  = hardlim(W.dot(x[i].reshape(-1,1)) + b).reshape(-1)
#pr = pr.argmax(0)
pr = bin2dec(pr)
#%% Evaluate by Confusion matrix:
P = bin2dec(y)
print(confusion_matrix(pr, P))
#%% plot output:
#plt.figure()
#plt.scatter(x[:,0], x[:,1], c=['red']*50+['orange']*50)
plt.title('predicted')
plot_hyperplane(W[0], b[0], -1, 1, 'k--', 1)
plot_hyperplane(W[1], b[1], -1, 1, 'k--', 2)
#plot_hyperplane(W[2], b[2], -1, 1, 'k--', P)

h = 0.01;xx, yy = np.meshgrid(np.arange(-1, 1.1, h), np.arange(-1, 1.1, h))
Z = predictovermesh(W, b, xx, yy)
plt.contourf(xx, yy, Z, cmap=cm.RdBu, alpha=.8)

plt.scatter(x[:,0], x[:,1], c=['red']*25+['orange']*25+['green']*25+['blue']*25)


plt.axis([-1,1.1,-1,1.1])