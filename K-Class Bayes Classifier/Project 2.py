#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 22:30:24 2017

@author: jrosaler
"""

#BAYES CLASSIFIER

import pandas as pd
import numpy as np

#REVISE BEFORE SUBMISSION 
#Split dataset into training and test sets. 
df = pd.read_csv('Iris.csv', header = None)
df = df.sample(frac=1).reset_index(drop=True)


#Replace strings with integers: setosa = 0, versicolor = 1, virginica = 2
df = df.replace('Iris-setosa', 0)
df = df.replace('Iris-versicolor', 1)
df = df.replace('Iris-virginica', 2)

df_train = df.loc[0:119, 0:4]
df_test = df.loc[120:150, 0:4]

X_train = df_train.loc[ : , 0:3]
y_train = df_train.loc[ : , 4]

X_test = df_test.loc[ : , 0:3]
y_test = df_test.loc[ : , 4]


#Set P(Y=y) estimates
num_setosa = len(y_train[y_train == 0].index)
num_versicolor = len(y_train[y_train == 1].index)
num_virginica = len(y_train[y_train == 2].index)
total = len(y_train.index)

pi_setosa = num_setosa/total
pi_versicolor = num_versicolor/total
pi_virginica = num_virginica/total

#Set P(x|y) distributions
#Specifically, set conditional class mean mu_y and covariance sigma_y
df_train_setosa = df_train[df_train.loc[ : , 4] == 0]
X_train_setosa = df_train_setosa.ix[ : , 0:3]

df_train_versicolor = df_train[df_train.loc[ : , 4] == 1]
X_train_versicolor = df_train_versicolor.ix[ : , 0:3]

df_train_virginica = df_train[df_train.loc[ : , 4] == 2]
X_train_virginica = df_train_virginica.ix[ : , 0:3]

mu_setosa = X_train_setosa.mean(axis = 0)
mu_versicolor = X_train_versicolor.mean(axis = 0)
mu_virginica = X_train_virginica.mean(axis = 0)

std_setosa = X_train_setosa.std(axis = 0)
std_versicolor = X_train_versicolor.std(axis = 0)
std_virginica = X_train_virginica.std(axis = 0)

"""
#Define standardized values in X_trained (x-mu)/std
#X_train_setosa_norm = X_train_setosa.values - mu_setosa.values
X_setosa_norm_array = (X_train_setosa.values - mu_setosa.values)/std_setosa.values
X_setosa_norm = pd.DataFrame(X_setosa_norm_array)

X_versicolor_norm_array = (X_train_versicolor.values - mu_versicolor.values)/std_versicolor.values
X_versicolor_norm = pd.DataFrame(X_versicolor_norm_array)

X_virginica_norm_array = (X_train_virginica.values - mu_virginica.values)/std_virginica.values
X_virginica_norm = pd.DataFrame(X_virginica_norm_array)
"""
#Define empirical covariance for each class
n_feat = len(X_train.columns)

n_setosa = len(X_train_setosa.index)
n_versicolor = len(X_train_versicolor.index)
n_virginica = len(X_train_virginica.index)

X_setosa_array = X_train_setosa.values
X_versicolor_array = X_train_versicolor.values
X_virginica_array = X_train_virginica.values

mu_setosa_array = mu_setosa.values
mu_versicolor_array = mu_versicolor.values
mu_virginica_array = mu_virginica.values

sigma_setosa = np.zeros((len(X_train_setosa.columns), len(X_train_setosa.columns)))
for i in range(n_setosa):
    sigma_setosa = sigma_setosa + np.outer(X_setosa_array[i][:] - mu_setosa_array, X_setosa_array[i] - mu_setosa_array)
sigma_setosa = sigma_setosa/n_setosa

sigma_versicolor = np.zeros((len(X_train_versicolor.columns), len(X_train_versicolor.columns)))
for i in range(n_versicolor):
    sigma_versicolor = sigma_versicolor + np.outer(X_versicolor_array[i][:] - mu_versicolor_array, X_versicolor_array[i] - mu_versicolor_array)
sigma_versicolor = sigma_versicolor/n_versicolor

sigma_virginica = np.zeros((len(X_train_virginica.columns), len(X_train_virginica.columns)))
for i in range(n_virginica):
    sigma_virginica = sigma_virginica + np.outer(X_virginica_array[i][:] - mu_virginica_array, X_virginica_array[i] - mu_virginica_array)
sigma_virginica = sigma_virginica/n_virginica


"""
#print(sigma_setosa)

sigma_versicolor = np.zeros((len(X_train_versicolor.columns), len(X_train_versicolor.columns)))
n_versicolor = len(X_train_versicolor.index)
for i in range(n_versicolor):
    sigma_versicolor = sigma_versicolor + np.outer(X_train_versicolor.ix[i] - mu_versicolor, X_train_versicolor.ix[i] - mu_versicolor)
sigma_versicolor = sigma_versicolor/(n_versicolor)
#print(sigma_versicolor)

sigma_virginica = np.zeros((len(X_train_virginica.columns), len(X_train_virginica.columns)))
n_virginica = len(X_train_virginica.index)
for i in range(n_virginica):
    sigma_virginica = sigma_virginica + np.outer(X_train_virginica.ix[i] - mu_virginica, X_train_virginica.ix[i] - mu_virginica)
sigma_virginica = sigma_virginica/(n_virginica)
#print(sigma_virginica)
"""


"""
n_feat = len(X_setosa_norm.columns)

sigma_setosa = np.zeros((len(X_setosa.columns), len(X_setosa_norm.columns)))
n_setosa = len(X_setosa_norm.index)
for i in range(n_setosa):
    sigma_setosa = sigma_setosa + np.outer(X_setosa_norm.ix[i], X_setosa_norm.ix[i])
sigma_setosa = sigma_setosa/(n_setosa)
#print(sigma_setosa)

sigma_versicolor = np.zeros((len(X_versicolor_norm.columns), len(X_versicolor_norm.columns)))
n_versicolor = len(X_versicolor_norm.index)
for i in range(n_versicolor):
    sigma_versicolor = sigma_versicolor + np.outer(X_versicolor_norm.ix[i], X_versicolor_norm.ix[i])
sigma_versicolor = sigma_versicolor/(n_versicolor)
#print(sigma_versicolor)

sigma_virginica = np.zeros((len(X_virginica_norm.columns), len(X_virginica_norm.columns)))
n_virginica = len(X_virginica_norm.index)
for i in range(n_virginica):
    sigma_virginica = sigma_virginica + np.outer(X_virginica_norm.ix[i], X_virginica_norm.ix[i])
sigma_virginica = sigma_virginica/(n_virginica)
#print(sigma_virginica)
"""


"""
#normalize test set
mu_test = X_test.mean(axis = 0)
std_test = X_test.std(axis = 0)

X_test_norm_array = (X_test.values - mu_test.values)/std_test.values
X_test_norm = pd.DataFrame(X_test_norm_array)
"""

#classify rows in X_test


pi = [pi_setosa, pi_versicolor, pi_virginica]
mu = [mu_setosa_array, mu_versicolor_array, mu_virginica_array]
sigma = [sigma_setosa, sigma_versicolor, sigma_virginica]
sigma_inv = [np.linalg.inv(sigma_setosa), np.linalg.inv(sigma_versicolor), np.linalg.inv(sigma_virginica)]

n_test = len(X_test.index)
d_test = len(X_test.columns)
n_classes = 3

X_test_array = X_test.values

"""
print(X_test_array)
for i in range(n_test):
    print(X_test_array[i])
    print(mu[2])
    print(X_test_array[i] - mu[2])
    print(sigma_inv[2])
    quad_prod = np.dot(np.dot(X_test_array[i] - mu[2], sigma_inv[2]), X_test_array[i] - mu[2])
    print(quad_prod)
    print("")
"""    


#create array of classification probabilities for each row/observation in X_test)
P_y_x = np.zeros((n_test, n_classes))
for i in range(n_test):  #Use Bayes' Rule, P(y|x) = P(y)P(x|y)/P(x), find for each of three classes y. Initialize to zeros. 
    for j in range(n_classes):
        P_y_x[i][j] = pi[j]*np.exp(-.5 * np.dot(np.dot(X_test_array[i] - mu[j],sigma_inv[j]),X_test_array[i] - mu[j]))
        #P_y_x[i][j] = pi[j]*np.exp(.5*np.dot(np.dot((X_test.iloc[i][:] - mu[j]),sigma[j]),(X_test.iloc[i][:] - mu[j]))) #How to calculate P(x)?
        P_y_x[i] = P_y_x[i]/np.sum(P_y_x[i])
        #print(P_y_x[i])
        #print("")
    #P_y_x[i] = P_y_x[i]/np.sum(P_y_x[i])
    print(P_y_x)



#print(df_test)



