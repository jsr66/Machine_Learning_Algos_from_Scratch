#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 22:16:45 2017

@author: jrosaler
"""

import numpy as np
import sys

""" CONVERT FILE INTO CORRECT FORM
ratings = np.genfromtxt ('ratings_raw.csv', delimiter=",")
ratings = ratings[1:,0:3]
np.savetxt("ratings.csv", ratings, delimiter=",")
"""

ratings = np.genfromtxt(sys.argv[1], delimiter = ",")
ratings = ratings.astype(float)
#print(ratings.shape)
#print(type(ratings[105][2]))
#dfy_test = pd.read_csv('y_test.csv', header = None)

d = 5
sigma_2 = .1
l = 2
#N_users = len(np.unique(ratings[:,0]))
N_users = int(max(ratings[:,0]) - min(ratings[:,0]) + 1)
N_objects = int(max(ratings[:,1]) - min(ratings[:,1]) + 1)
N_entries = ratings.shape[0]
#print(N_users)
#print(N_objects)
#print(N_entries)
print("Ratings array imported.")

# Fill M_ij with ratings of ith user and jth object, zero in entries not included in file
M_init = np.zeros((N_users, N_objects))

for i in range(N_entries):
    user = int(ratings[i,0])
    obj = int(ratings[i,1])
    M_init[user-1,obj-1] = ratings[i,2]

print("M filled.")

#Truncate ratings matrix since it is too big for the algo/comp
M =  M_init[:,0:700 ] 
N_objects = 700 #redefine N_objects for truncated M

#Initialize v for each object. Each v_j corresponds to a column of v
v = np.zeros((d, N_objects)) #produces dxN_objects matrix

for i in range(N_objects):
    v[:,i] = np.random.normal(0, sigma_2, d) 
    print(v[:,i].astype(str))
 
print("v initialized.")
#print("v.shape: " + str(v.shape))
#print(str(np.dot(v.T,v)))


#Update u_i from initialized v_j's and M
#u = np.zeros((N_users, d))

def objective(M, u, v): 
    L = - 1/(2*sigma_2)*np.sum(np.square(M - np.dot(u.T,v))) - (l/2)*(np.trace(np.dot(u.T,u))) - (l/2)*(np.trace(np.dot(v.T,v)))
    #print("in objective, L: " + str(L))
    return L

N_iter = 50

#Declare array to hold 50 values of objective function. 


Objective_function = []
u = np.zeros((d, N_users)) #initialize matrix of user d-vectors, one per column, with N_users columns, making a dxN_users maatrix 

for i in range(N_iter):
    print(i)
    
    Mv = np.dot(M,v.T) #produces ixd matrix
    D =  l*np.identity(d) + np.dot(v, v.T) #produces dxd matrix
    D_inv = np.linalg.inv(D) #produces dxd matrix
    u = np.dot(D_inv, Mv.T) #produces dxN_users matrix
    #print(u.shape)
    #print(M.shape)

    Obj_iter = objective(M, u, v)
    Objective_function.append(Obj_iter)
    print("objective function: " + str(Obj_iter))
    
    Mu = np.dot(u,M) #gives a dxN_objects matrix
    E = l*np.identity(d) + np.dot(u, u.T)
    E_inv = np.linalg.inv(E)
    v = np.dot(E_inv, Mu) #gives a dxN_objects matrix

#
print("u: " + str(u))
print("v: " + str(v))
      
Objective_function = np.array(Objective_function) #convert list to array
np.savetxt("objective.csv" , Objective_function) #write array to .csv file

print_iterations = [10, 25, 50]

for num in print_iterations:
    np.savetxt("U-" + str(num) + ".csv", u.T, delimiter = ",")
    np.savetxt("V-" + str(num) + ".csv", v.T, delimiter = ",")
    
    
    
    





 
