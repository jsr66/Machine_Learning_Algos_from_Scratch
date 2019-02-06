#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 15:53:23 2017

@author: jrosaler
"""
import numpy as np
import pandas as pd
import sys

#GOOD WORKING CODE, RUN IN SPYDER OR OTHER ENVIRONMENT RATHER THAN COMMAND LINE - BASICALLY SAME AS hw2_classification.py, but run from
#environment instead of mac terminal. 

"""
#CREATE X_TRAIN, Y_TRAIN, X_TEST, Y_TEST FROM IRIS.CSV, RUN ONCE
df = pd.read_csv('Iris.csv', header = None)
df = df.sample(frac=1).reset_index(drop=True)

#Replace strings with integers: setosa = 0, versicolor = 1, virginica = 2
df = df.replace('Iris-setosa', int(0))
df = df.replace('Iris-versicolor', int(1))
df = df.replace('Iris-virginica', int(2))

#split into training and test sets
df_train = df.loc[0:119, 0:4]
df_test = df.loc[120:150, 0:4]

#split training and test dataframes into covariates X and labels y
X_train = df_train.loc[ : , 0:3]
y_train = df_train.loc[ : , 4]

X_test = df_test.loc[ : , 0:3]
y_test = df_test.loc[ : , 4]

np.savetxt("X_train.csv", X_train, delimiter = ",")
np.savetxt("y_train.csv", y_train, delimiter = ",")
np.savetxt("X_test.csv", X_test, delimiter = ",")
np.savetxt("y_test.csv", y_test, delimiter = ",")
"""



#import data frames from csv files
df_X_train = pd.read_csv('X_train.csv', header = None)
df_y_train = pd.read_csv('y_train.csv', header = None)
df_X_test = pd.read_csv('X_test.csv', header = None)

#determine number of classes
num_classes = len(df_y_train.iloc[:][0].unique())
num_feat = len(df_X_train.columns)

#merge df_X_train and df_y_train to select class-specific dataframes
df_train = pd.merge(df_X_train, df_y_train, left_index=True, right_index=True, how='inner')
num_train = len(df_train.index) #number of training observations

df_class = [] #make list of dataframes specific to each class
array_class = [] #make list of arrays specific to each class
X_array_class = []
num_class = [] #make list of number of observations in each class
pi_class = [] #make list of values P(y), unconditional probability for each class
mu_class = [] #make list of class-specific means
sigma_class = [] #make list of class-specific covariance matrices

#Set df_class
for i in range(num_classes):
    df_class.append(df_train[df_train.iloc[:,4] == i])
#Set array_class
for i in range(num_classes):
    array_class.append(df_class[i].values)
#Set X_array_class
for i in range(num_classes):
    X_array_class.append(array_class[i][:,0:4])
#Set num_class    
for i in range(num_classes):
    num_class.append(len(df_class[i].index))
#Set pi_class
for i in range(num_classes):
    pi_class.append(num_class[i]/num_train)
#Set mu_class
for i in range(num_classes):
    mu_class.append(np.mean(X_array_class[i], axis = 0))
#Set sigma_class - FIX!!!!!!!!!!!
for i in range(num_classes):
    sigma_i = np.zeros((num_feat, num_feat))
    for j in range(num_class[i]): #sum over observations
        sigma_i = sigma_i + (1/num_class[i])*np.outer(X_array_class[i][j,:] - mu_class[i], X_array_class[i][j,:] - mu_class[i])
    sigma_class.append(sigma_i)

"""
for i in range(len(sigma_class)):
    print(sigma_class[i])
    print("")

for i in range(num_classes):
    print(i)
    print("Mu" + str(mu_class[i]))
    sigma_i = np.zeros((num_feat, num_feat))
    for j in range(num_class[i]): #sum over observations
        print(X_array_class[i][j,:])
        print(X_array_class[i][j,:] - mu_class[i])
    print("")
"""   
num_test = len(df_X_test.index) #number of observations in X_test

#fill array with class probabilities for each observation
P_y_x = np.zeros((num_test, num_classes))

#convert df_X_test to array for linear algebra manipulations
X_test = df_X_test.values

#fill array P_y_x with class probabilities for each observation in X_test
for i in range(num_test): #for each observation/row in X_test, calculate probability of its being in each class P(y|x) ~ P(y)P(x|y)
    #print("i= " + str(i))
    for j in range(num_classes):
        #print("j= " + str(j))
        sigma_inv = np.linalg.inv(sigma_class[j])
        quad_prod = np.dot(np.dot(X_test[i] - mu_class[j], sigma_inv), X_test[i] - mu_class[j])
        P_y_x[i][j] = pi_class[j]*(1/np.linalg.det(sigma_class[j]))*np.exp(-.5 * quad_prod)
    P_y_x[i] = P_y_x[i]/np.sum(P_y_x[i])
    #print(np.sum(P_y_x[i]))
        #print(np.linalg.det(sigma_class[j]))
    #print("")
        #P_y_x[i][j] = pi_class[i]*
        
np.savetxt("probs_test.csv", P_y_x, delimiter = ",")
  
classifier = np.zeros(num_test)
classifier.fill(9999)
  
for i in range(num_test):
    print(np.argmax(P_y_x[i]))
    classifier[i] = np.argmax(P_y_x[i])
    
print(classifier)
    
        
    
    

    

 
  
   






