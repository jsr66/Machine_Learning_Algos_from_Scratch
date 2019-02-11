import numpy as np
from getLogLikelihood import getLogLikelihood

def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).

    #####Insert your code here for subtask 6c#####
    N = gamma.shape[0] #number of examples
    K = gamma.shape[1] #number of classes
    D = X.shape[1] #number of features
    
    N_class = np.zeros((K)) 
    
    weights = np.zeros((K))
    means = np.zeros((K, D))
    covariances = np.zeros((D, D, K))
    
    #Set class weights
    for k in range(K):
        for n in range(N):
            N_class[k] += gamma[n][k]
        weights[k] = N_class[k] / N   
    
    #Set class means
    for k in range(K):
        mu_k = 0
        for n in range(N):
            x_n = X[n, :]
            N_k = N_class[k]
            mu_k += gamma[n][k] * x_n
        mu_k = mu_k / N_k
        means[k, :] = mu_k
    
    #Set class covariance matrices
    for k in range(K):
        for n in range(N):
            x_n = X[n, :]
            mu_k = means[k, :]
            N_k = N_class[k]
            covariances[:, :, k] += gamma[n][k] * np.outer((x_n - mu_k),(x_n - mu_k))
        covariances[:, :, k] = covariances[:, :, k] / N_k
            
    logLikelihood = getLogLikelihood(means, weights, covariances, X)
    
    return weights, means, covariances, logLikelihood
