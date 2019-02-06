#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 15:53:23 2017

@author: jrosaler
"""
import numpy as np
import sys

#RUN FROM TERMINAL WITH COMMAND 
#python hw2_classification.py X_train.csv y_train.csv X_test.csv

#import data frames from csv files
X_train = np.genfromtxt(sys.argv[1], delimiter = ",")
y_train = np.genfromtxt(sys.argv[2], delimiter = ",")
#print("X_train: " + str(X_train))
#print("y_train: " + str(y_train))
num_obs = y_train.shape[0]
y_train = np.reshape(y_train, (num_obs, 1))

X_test = np.genfromtxt(sys.argv[3], delimiter = ",")
#print("X_test: " + str(X_test))




""""""""""""""""""""""""""#COMMENT OUT WHEN DONE TESTING
""""""""""""""""""""""""""
"""
y_test = np.genfromtxt("y_test.csv", delimiter = ",")
num_obs_test = y_test.shape[0]
y_test = np.reshape(y_test, (num_obs_test, 1))
"""
""""""""""""""""""""""""""
""""""""""""""""""""""""""



#determine number of classes
num_classes = np.unique(y_train).shape[0]
print("num_classes: " + str(num_classes))
num_feat = len(X_train[0,:])
#print("num_feat: " + str(num_feat))



#merge X_train and y_train to select class-specific dataframes
train = np.concatenate((X_train, y_train), axis = 1)
#print("train array: " + str(train))
#print(train.shape)
num_train = num_obs #number of training observations
#print("num_train: " + str(num_train))


array_class = [] #make list of arrays specific to each class
X_array_class = [] #make list specifically covariate arrays specific to each class
num_class = [] #make list of number of observations in each class
pi_class = [] #make list of values P(y), unconditional probability for each class
mu_class = [] #make list of class-specific means
sigma_class = [] #make list of class-specific covariance matrices

#Set array_class
for i in range(num_classes):
    array_i = train[train[:,-1] == i]
    #print("in set array_class, array_i: " + str(array_i))
    array_class.append(array_i)
#print("")
#Set X_array_class
for i in range(num_classes):
    X_array_i = array_class[i][:,:-1]
    #print("in X_array_class, X_array_i: " + str(X_array_i))
    X_array_class.append(X_array_i)
#print("")
#Set num_class    
for i in range(num_classes):
    num_class_i = array_class[i][:,0].shape[0]
    #print("in num_class, num_class_i: " + str(num_class_i))
    num_class.append(num_class_i)
#print("")
#Set pi_class
for i in range(num_classes):
    #print("in set pi_class, i: " + str(i))
    #print("in set pi_class, num_class[i]: " + str(num_class[i]))
    #print("in set pi_class, num_train: " + str(num_train))
    pi = num_class[i]*1.0/(num_train*1.0)
    #print("in set pi_class, pi: " + str(pi))
    pi_class.append(pi)
#print("")
#Set mu_class
for i in range(num_classes):
    mu_i = np.mean(X_array_class[i], axis = 0)
    #print("in set mu_class, mu_i: " + str(mu_i))
    mu_class.append(mu_i)
#print("")
#Set sigma_class - FIRST PROBLEM IS HERE!!!
for i in range(num_classes):
    #print("in set_sigma, i = : " + str(i))
    sigma_i = np.zeros((num_feat, num_feat))
    for j in range(num_class[i]): #sum over observations in class i in X_train
        #print("in Set_sigma j= " + str(j))
        #add to class_specific covariance matrix
        #print("(1/(num_class[i]*1.0)): " + str((1.0/(num_class[i]*1.0))))
        sigma_i = sigma_i + (1.0/(num_class[i]*1.0))*np.outer(X_array_class[i][j,:] - mu_class[i], X_array_class[i][j,:] - mu_class[i])
    sigma_class.append(sigma_i)
    #print("in set_sigma, sigma_i = : " + str(sigma_i))

"""
#TEST

print("")
for arr in array_class:
    print("Checking array_class: ")
    print(arr)

print("")
for arr in X_array_class:
    print("Checking X_array_class: ")
    print(arr)

print("")
for sigma in sigma_class:
    print("Checking sigma_class: ")
    print(sigma)

print("")
for p in pi_class:
    print("Checking pi_class: ")
    print(p)
    
print("")
for mu in mu_class:
    print("Checking mu_class: ")
    print(mu)

print("")
for num in num_class:
    print("Checking num_class: ")
    print(num)
"""




num_test = X_test[:,0].shape[0] #number of observations in X_test
print("num_test: " + str(num_test))
print("X_test: " + str(X_test))
#print("")
#print("Checking num_test: " + str(num_test))
#fill array with class probabilities for each observation
P_y_x = np.zeros((num_test, num_classes))

#Sigma_class, mu_class, X_test, all checked with previous successful implementation

#EARLY MARCH CODE
#fill array P_y_x with class probabilities for each observation in X_test
for i in range(num_test): #for each observation/row in X_test, calculate probability of its being in each class P(y|x) ~ P(y)P(x|y)
    #print("i= " + str(i))
    #print("num_test observation: " + str(i))
    for j in range(num_classes):
        #print("num_test observation: " + str(i))
        #print("class-specific inverse covariance matrix: " + str(j))
        sigma_inv = (np.linalg.inv(sigma_class[j])).astype('float')
        quad_prod = np.dot(np.dot(X_test[i] - mu_class[j], sigma_inv), X_test[i] - mu_class[j])
        P_y_x[i][j] = pi_class[j]*float(1/float(np.linalg.det(sigma_class[j])))*np.exp(-.5 * quad_prod)
    #print("Checking sigma_inv: " + str(sigma_inv))
    P_y_x[i] = P_y_x[i]/np.sum(P_y_x[i])
    print("")
#print(np.sum(P_y_x[i]))
#print(np.linalg.det(sigma_class[j]))
#print("")
#P_y_x[i][j] = pi_class[i]*

np.savetxt("probs_test.csv", P_y_x, delimiter = ",")


"""
#APR 9 CODE
#fill array P_y_x with class probabilities for each observation in X_test
for i in range(num_test): #for each observation/row in X_test, calculate probability of its being in each class P(y|x) ~ P(y)P(x|y)
    #print("i= " + str(i))
    #print("num_test observation: " + str(i))
    for j in range(num_classes):
        #print("num_test observation i: " + str(i))
        #print("class j: " + str(j))
        #print("pi_class[j]: " + str(pi_class[j]))
        sigma_inv = (np.linalg.inv(sigma_class[j])).astype('float')
        quad_prod = np.dot(np.dot(X_test[i] - mu_class[j], sigma_inv), X_test[i] - mu_class[j])
        P_y_x[i,j] = pi_class[j]*float(1/float(np.linalg.det(sigma_class[j])))*np.exp(-.5 * quad_prod)
        #print("Checking P_y_x[i][j]: " + str(P_y_x[i,j]))
        P_y_x[i,:] = (P_y_x[i,:]*1.0)/(np.sum(P_y_x[i,:])*1.0)
        #print("Checking P_y_x[i][j]: " + str(P_y_x[i,j]))
    #print("")

        
np.savetxt("probs_test.csv", P_y_x, delimiter = ",")
"""

"""
#PRINT CLASS PREDICTIONS
classifier = np.zeros(num_test)
classifier.fill(9999)

for i in range(num_test):
    #print(np.argmax(P_y_x[i]))
    classifier[i] = np.argmax(P_y_x[i])

#print(classifier)

#determine percentage of predictions that were correct by comparison y_test
correct = 0
#print(y_test)

for i in range(num_test):
    if(classifier[i] == y_test[i,0]):
        correct = correct + 1

print(correct/num_test)
"""

#PRINT CLASS PREDICTIONS
"""#COMMENT OUT WHEN DONE TESTING
"""
"""
classifier = np.zeros((num_test,1))
classifier.fill(9999)
  
for i in range(num_test):
    #print(np.argmax(P_y_x[i]))
    classifier[i] = np.argmax(P_y_x[i,:])
    
#print(classifier)
#print(classifier.shape)

#print("")
#determine percentage of predictions that were correct by comparison y_test
correct = 0
#print(y_test)
#print(y_test.shape)

for i in range(num_test):
    #print(i)
    #print(classifier[i,0])
    #print(y_test[i,0])
    #print(classifier[i,0] == y_test[i,0])
    if classifier[i,0] == y_test[i,0]:
        correct = correct + 1
  
print(correct/(num_test*1.0))
"""
        
    
    

    

 
  
   






