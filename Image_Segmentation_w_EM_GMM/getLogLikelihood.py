import numpy as np


def Gaussian(x, mu, Sigma):
    #mu: mean, 1xD array
    #Sigma: covariance matrix, DxD array
    #x: position, 1xD array
    #print("In Gaussian, Sigma.shape " + str(Sigma.shape))
    D = len(mu)
    y = (1./(((2*np.pi)**(D/2))*np.sqrt(np.linalg.det(Sigma)))) * np.exp(-.5*np.dot(x - mu, np.dot(np.linalg.inv(Sigma), x - mu)))
    return y

def getLogLikelihood(means, weights, covariances, X): # Log Likelihood estimation
#
# INPUT:
# means          :Mean for each Gaussian KxD
# weights      :Weight vector 1xK for K Gaussians
# covariances     :Covariance matrices for each gaussian DxDxK
# X  :Input data NxD  
# where N is number of data points
# D is the dimension of the data points
# K is number of gaussians: 
#
# OUTPUT:
# logLikelihood : log-likelihood 
    N = X.shape[0]
    K = len(weights)
    logL = 0 #initialize likelihood function
    for n in range(N):
        x_n = X[n,:]
        p_xn = 0
        for i in range(K):
            mu_i = means[i]
            Sigma_i = covariances[:, :, i]
            p_xn += weights[i]*Gaussian(x_n, mu_i, Sigma_i)  
        logL += np.log(p_xn)
    return logL
