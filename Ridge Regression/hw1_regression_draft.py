#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#from pandas import *
import pandas as pd
import numpy as np
import sys

"""
Data import and cleaning
"""

Lambda = int(sys.argv[1])
Sigma2 = float(sys.argv[2])
dfX_train = pd.read_csv(sys.argv[3], header = None)
dfy_train = pd.read_csv(sys.argv[4], header = None)
dfX_test = pd.read_csv(sys.argv[5], header = None)
#dfy_test = pd.read_csv('y_test.csv', header = None)

print("lambda:" + str(Lambda))
print("sigma2:" + str(Sigma2))


X_train = dfX_train.as_matrix()
X_train = X_train.astype(np.float)
#print(type(X_train[234][0]))
#print(X_train.shape)
print("X_train" + str(X_train))

y_train = dfy_train.as_matrix()
y_train = y_train.astype(np.float)
print("y_train" + str(y_train))

X_test = dfX_test.as_matrix()
X_test = X_test.astype(np.float)
print("X_test" + str(X_test))

#y_test = dfy_test.as_matrix()
#y_test = y_test.astype(np.float)

print("X_train shape:" + str(X_train.shape))
print("y_train shape:" + str(y_train.shape))
print("X_test shape:" + str(X_test.shape))


#X_train = float(sys.arg[3])
#y_train = float(sys.arg[4])
#X_test = float(sys.arg[5])




"""
PART 1
"""


def ridge_regression(X, y, L):
    XTy = np.dot(X.T, y)
    D =  L*np.identity(X.shape[1]) + np.dot(X.T, X)
    print("in ridge_regression, D: " + str(D))
    D_inv = np.linalg.inv(D)
    print("in ridge_regression, D_inv: " + str(D_inv))
    return np.dot(D_inv, XTy)

#PREDICT ON TEST DATA   
w_rr = ridge_regression(X_train,y_train, Lambda)
#print('{0:.2f}'.format(w_rr))
#print("wRR:" + str(w_rr))
np.savetxt("wRR_" + str(Lambda) + ".csv", w_rr)

#COMPARE Y_PRED TO Y_TEST
#y_pred = np.dot(X_test, w_rr)
#print(y_test)
#print(y_pred)
#print((y_test - y_pred)/y_test)



"""
PART 2
"""


#Output quadratic product of covariate vector for a single observation with posterior covariance matrix
def quad_prod(x, sigma):
    return np.dot(np.dot(x.T, sigma), x)

#Find index of observation that produces the maximum quadratic product, and therefore predictive uncertainty
def find_max_x(sigma, X): #returns index of observation with largest uncertainty in predictive distribution, \
#given dxd covariance matrix for posterior distribution, and an nXd matrix of covariates X. 
    quad_prod_max = quad_prod(X[0], sigma)
    max_index = 0 #row index of observation that maximizes quadratic product, which maximizes predictive distribution uncertainty
    for i in range(X.shape[0]):
        if quad_prod(X[i], sigma) > quad_prod_max:
            max_index = i
            quad_prod_max = quad_prod(X[i], sigma)
    return max_index                                         

x_indices = []

sigma = np.linalg.inv(Lambda*np.identity(X_train.shape[1]) + (1/Sigma2)*np.dot(X_train.T, X_train))

#print(find_max_x(sigma, X_train))
#print(X_train.shape)

i=0

while i < 10:
    #set temporary X matrix to delete rows as we sequentially delete observations that maximize predictive uncertainty
    X_temp = X_test
    #find index of observation that maximizes predictive uncertainty
    index = find_max_x(sigma, X_temp)
    #add this to the set of indices that maximize predictive uncertainty, in order from greatest to least
    x_indices.extend([index+1])
    #update posterior covariance matrix with this max observation
    sigma = np.linalg.inv(np.linalg.inv(sigma) + (1/Sigma2)*np.outer(X_temp[index],X_temp[index]))
    #drop observation from covariate matrix, to find max among remaining observations in next cycle of while loop
    X_temp[index] = np.zeros(X_temp.shape[1])
    #print("X_temp shape:" + str(X_temp.shape))
    #increment index 
    i = i+1
  
print(x_indices)

np.savetxt("active_" + str(Lambda) + "_"+ str(Sigma2) + ".csv", x_indices, delimiter = ",", newline = ",", fmt = "%d")
