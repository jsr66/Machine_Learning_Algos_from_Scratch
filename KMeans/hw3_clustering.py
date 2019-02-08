#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 22:16:45 2017

@author: jrosaler
"""

import numpy as np
import sys

"""
#CONVERT FILE INTO ARRAY, OUTPUT ARRAY
X = np.genfromtxt(sys.argv[1], delimiter = ",")
X = X.astype(float)
X = X[1:,1:] #remove headers
np.savetxt("X.csv", X, delimiter=",")
"""


X = np.genfromtxt(sys.argv[1], delimiter = ",")
num_obs = len(X[:,0])
num_features = len(X[0,:])
#print("num_features: " + str(num_features))
#print(X)

"""""""" #CHANGE ON SUBMISSION
num_clusters = 5
""""""""
num_iterations = 1

#Initialize centroids to  randomly chosen data points
random_indices = np.random.choice(range(num_obs), size = num_clusters, replace = False)
centroids_init = np.take(X, random_indices, axis = 0)
#print(centroids_init)






#K-MEANS - GOOD
#Output one file of centroids for each iteration. Each file contains five rows, each row with a 5-D centroid for its cluster

#initialize centroids to num_clusters data points
centroids_iter = centroids_init

for i in range(num_iterations):
    #print("iteration: " + str(i))
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
        #print("centroid")
        #print(centroid)
    np.savetxt("centroids-" + str(i+1) + ".csv", centroids_iter, delimiter = ",")




'''

#GMM EM soft clustering - STILL BUGGY
pi_init = (1.0/(num_clusters*1.0))*np.ones((num_clusters, 1))
mu_init = centroids_init
Sigma_init = np.identity(num_features)
phi_init = np.empty((num_obs, num_clusters))

pi_iter = pi_init
mu_iter = mu_init
Sigma_iter = [Sigma_init for i in range(num_clusters)]
phi_iter = phi_init
n_cluster = np.zeros(num_clusters)

for i in range(num_iterations):
    #E-Step
    for j in range(num_obs):
        #print("j (obs): " + str(j))
        numerators = np.ones(num_clusters)
        denominator = 0
        for k in range(num_clusters):
            #print("k (cluster): " + str(k))
            sigma_inv = np.linalg.inv(Sigma_iter[k])
            numerators[k] = pi_iter[k,0]*(  (1.0/((2*np.pi*np.linalg.det(Sigma_iter[k]))**(.5)))  * np.exp(-.5*np.dot(np.dot((X[j,:] - mu_iter[k,:]).T, sigma_inv), X[j,:] - mu_iter[k,:])))
            #print("numerators[k]: " + str(numerators[k]))
            denominator = denominator + numerators[k]
        #print("denominator: " + str(denominator))
        for k in range(num_clusters):
            #print("j: " + str(j))
            #print("k: " + str(k))
            phi_iter[j,k] = (numerators[k]*1.0)/(denominator*1.0)
            #print("phi_iter[j,k]: " + str(phi_iter[j,k]))
    #M-Step
    for k in range(num_clusters):
        n_cluster[k] = np.sum(phi_iter[:,k])
        #print("phi_iter[:,k]: " + str(phi_iter[:,k]))
        #print("j: ")
        #print("k: ")
        print("n_cluster[k]: " + str(n_cluster[k]))
        pi_iter[k] = (n_cluster[k]*1.0)/(num_obs*1.0)
        #print("n_cluster[k]: " + str(n_cluster[k]))
        mu_iter[k] = (1.0/n_cluster[k])*np.dot(phi_iter[:,k].T, X)
        Sigma = np.zeros((num_features, num_features))
        for j in range(num_obs):
            Sigma = Sigma + (1.0/n_cluster[k]) * phi_iter[j,k] * np.dot(X[j,:] - mu_iter[k] , (X[j,:] - mu_iter[k]).T )
        Sigma_iter[k] = Sigma

    #Save to file
    np.savetxt("pi-" + str(i+1) + ".csv", pi_iter, delimiter = ",")
    np.savetxt("mu-" + str(i+1) + ".csv", mu_iter, delimiter = ",")
    for k in range(num_clusters):
        np.savetxt("Sigma-" + str(k+1) + "-" + str(i+1) + ".csv", Sigma_iter[k], delimiter = ",")


'''



 
