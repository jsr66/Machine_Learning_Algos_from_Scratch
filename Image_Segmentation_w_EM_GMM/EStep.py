import numpy as np
from getLogLikelihood import getLogLikelihood, Gaussian


def EStep(means, covariances, weights, X):
    #INPUT
    #means:     Mean for each Gaussian KxD
    #covariances:    Covariance matrices for each Gaussian DxDxK
    #weights:     Weight vector 1xK for K Gaussians
    #X:     Input data NxD
    #OUTPUT
    #logLikelihood:    Log-likelihood (a scalar).
    #gamma:   NxK matrix of responsibilities for N datapoints and K Gaussians.
    #EXPLANATION:    Expectation step of the EM Algorithm. N is number of data points, D is the dimension of the data
    #points, K is number of Gaussians
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
        #Compute responsibilities gamma
        for k in range(K):
            mu_k = means[k]
            Sigma_k = covariances[:, :, k]
            gamma_nk = weights[k]*Gaussian(x_n, mu_k, Sigma_k)/Sum_n
            gamma[n][k] = gamma_nk
    logLikelihood = getLogLikelihood(means, weights, covariances, X)
    return [logLikelihood, gamma]
