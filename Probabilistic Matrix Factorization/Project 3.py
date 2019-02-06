#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 22:16:45 2017

@author: jrosaler
"""

import numpy as np
import sys

""" CONVERT FILE INTO ARRAY, OUTPUT ARRAY
"""
"""
X = np.genfromtxt(sys.argv[1], delimiter = ",")
X = X.astype(float)
np.savetxt("X.csv", X, delimiter=",")
"""
X = np.genfromtxt(sys.argv[1], delimiter = ",")
X = X[1:,1:] #remove headers
num_obs = len(X[:,0])
#print(X)

"""""""" #CHANGE ON SUBMISSION
num_clusters = 15
num_features = 2
""""""""
num_iterations = 10

#Initialize centroids to  randomly chosen data points
random_indices = np.random.choice(range(num_obs), size = num_clusters, replace = False)
centroids_init = np.take(X, random_indices, axis = 0)
#print(centroids_init)

#K-MEANS
#Output one file of centroids for each iteration. Each file contains five rows, each row with a 5-D centroid for its cluster

#initialize centroids to num_clusters data points
centroids_iter = centroids_init

for i in range(num_iterations):
    print("iteration: " + str(i))
    centroid_assignments = np.empty((num_obs,1))
    #Set centroid assignments for each centroid
    for j in range(num_obs):
        min = 10000 #set comparison value to something very large
        for k in range(num_clusters): #check for each centroid whether distance to that centroid is smaller
            if(np.linalg.norm(X[j,:] - centroids_iter[k,:]) < min): #if so, replace
                min = np.linalg.norm(X[j,:] - centroids_iter[k,:])
                centroid_assignments[j] = k
    #Update mean vectors for observations in each centroid
        #for each centroid k, select all rows of X assigned to centroid to k, take their average
    #print("X")
    #print(X)
    X_assigned = np.append(X,centroid_assignments, axis=1)
    #print("X_assigned")
    #print(X_assigned[X_assigned[:,-1] == 0])
    for k in range(num_clusters):
        X_centroid = X_assigned[X_assigned[:,-1] == k]
        #print(X_centroid)
        #X_centroid = X_centroid[:,:-1]
        centroid = np.mean(X_centroid, axis=0)
        centroids_iter[k,:] = centroid[:-1]
        #print("cluster")
        #print(k)
        print("centroid")
        print(centroid)
    np.savetxt("centroids-" + str(i), centroids_iter, delimiter = ",")


    
    
    
    





 
