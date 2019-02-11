import numpy as np
from getLogLikelihood import getLogLikelihood


def Gaussian(x, mu, Sigma):
    #mu: mean, 1xD array
    #Sigma: covariance matrix, DxD array
    #x: position, 1xD array
    y = (1./(np.sqrt(2*np.pi)*np.linalg.det(Sigma))) * np.exp(-.5*np.dot(x - mu, np.dot(np.linalg.inv(Sigma), x - mu)))
    return y


def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    #####Insert your code here for subtask 6b#####
    N = X.shape[0]
    K = len(weights)
    gamma = np.zeros((N,K))
    for n in range(N): 
        x_n = X[n,:]
        Sum_n = 0
        #Compute denominator Sum_n over different classes 
        for k in range(K):
            mu_k = means[k]
            Sigma_k = covariances[:, :, k]
            Sum_n += weights[k]*Gaussian(x_n, mu_k, Sigma_k)
        #Compute responsibilities
        for k in range(K):
            mu_k = means[k]
            Sigma_k = covariances[:, :, k]
            gamma_nk = weights[k]*Gaussian(x_n, mu_k, Sigma_k)/Sum_n
            gamma[n][k] = gamma_nk
    logLikelihood = getLogLikelihood(means, weights, covariances, X)
    return [logLikelihood, gamma]
