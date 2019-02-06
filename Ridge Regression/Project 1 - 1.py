# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#from pandas import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('ENB2012_data.csv', header = 0)

#CLEANING: GET RID OF EXTRA COLUMNS IN 'ENB2012_data.csv'
df = df.loc[:,'X1':'Y2']

#TRAINING DATA
dfx_train = df.loc[0:550, 'X1':'X8']
dfy_train = df.loc[0:550, 'Y1'] #train for only first output

#TEST DATA
dfx_test = df.loc[551:767, 'X1':'X8']
dfy_test = df.loc[551:767, 'Y1'] #test for only first output

#MATRIX MANIPULATIONS FOR RIDGE REGRESSION
#FIND W_RR
X = dfx_train.as_matrix()
X = np.c_[X, np.ones(X.shape[0])]
np.savetxt("X_train.csv", X, delimiter=',',fmt="%s")
#X = np.mat(X)
y = dfy_train.as_matrix()
np.savetxt("y_train.csv", y, delimiter=',',fmt="%s")
#
X_test = dfx_test.as_matrix()
X_test = np.c_[X_test, np.ones(X_test.shape[0])]
np.savetxt("X_test.csv", X_test, delimiter=',',fmt="%s")
#
y_test = dfy_test.as_matrix()
np.savetxt("y_test.csv", y_test, delimiter=',',fmt="%s")
#y = np.mat(y)

"""
PART 1
"""

#print(np.dot(X.T,y))
#np.mat(X.T)*np.mat(y)
D = np.linalg.inv(np.dot(X.T, X))
#print(11*np.identity(X.shape[1]))

L = .5 #lambda from l2 regularization
var = 2 #noise from likelihood


def ridge_regression(X, y, L):
    XTy = np.dot(X.T, y)
    D =  L*np.identity(X.shape[1]) + np.dot(X.T, X)
    D_inv = np.linalg.inv(D)
    return np.dot(D_inv, XTy)

#PREDICT ON TEST DATA
X_test = dfx_test.as_matrix()
y_test = dfy_test.as_matrix()    

w_rr = ridge_regression(X,y,L)
#y_pred = np.dot(X_test, w_rr)
np.savetxt("w_rr.csv", w_rr)

#COMPARE Y_PRED TO Y_TEST
#print(y_test)
#print(y_pred)
#print(y_test - y_pred)


"""
PART 2
"""
"""
def Gauss_non_norm(x, mu, sigma):
    sigma_inv = np.linalg.inv(sigma)
    prod = np.dot(np.dot((x - mu).T, sigma_inv), (x - mu))
    return np.exp(prod)
"""

""" GAUSS_NON_NORM TEST   
x = np.array([0,1,2])
mu = np.array([0,1,0])
sigma = np.array([[1,0,4],[0,2,0],[4,0,3]])

sigma_inv = np.linalg.inv(sigma)
prod = np.dot(np.dot((x - mu).T, sigma_inv), (x - mu))

print(np.dot((x - mu).T, sigma_inv))
print(Gauss_non_norm(x, mu, sigma))
"""

df_x = df.loc[:,'X1':'X8']
df_y = df.loc[:,'Y1']

def quad_prod(x, sigma):
    return np.dot(np.dot(x.T, sigma), x)

def find_max_x(sigma, df): #returns index of observation with largest uncertainty in predictive distribution, \
#given dxd covariance matrix for posterior distribution, and an nXd dataframe of covariates X. 
    quad_prod_max = quad_prod(df.ix[0], sigma)
    max_index = 0 #row index of observation that maximizes quadratic product, which maximizes predictive distribution uncertainty
    for i in range(len(df.index)):
        if quad_prod(df.ix[i], sigma) > quad_prod_max:
            max_index = i
            quad_prod_max = quad_prod(df.ix[i], sigma)
    return max_index                                         
    
x_indices = []
x_values = []

sigma = (1/L)*np.identity(len(df_x.columns)) # initial dXd covariance matrix of prior over w
#var_y = var #initial variance of predictive distribution
#prior = 1
#likelihood = Gauss_non_norm()
i=0

while i < 10:
    #set temporary X matrix to delete rows as we sequentially delete observations that maximize predictive uncertainty
    df_temp = df_x
    #find index of observation that maximizes predictive uncertainty
    index = find_max_x(sigma, df_x)
    #add this to the set of indices that maximize predictive uncertainty, in order from greatest to least
    x_indices.extend([index])
    #update posterior covariance matrix with this max observation
    sigma = np.linalg.inv(np.linalg.inv(sigma) + (1/var)*np.outer(df_x.ix[index],df_x.ix[index]))
    #drop observation from covariate matrix, to find max among remaining observations in next cycle of while loop
    df_temp = df_temp.drop(df.index[index]) 
    #increment index 
    i = i+1
  
print(x_indices)

