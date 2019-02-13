import numpy as np
from numpy.linalg import eig
from numpy import diag
from numpy import dot
from numpy.linalg import inv
from numpy import array
from numpy.linalg import eig



def regularize_cov(covariance, epsilon):
    #INPUT:
    #covariance:     DxD matrix
    #epsilon:     regulator, minimum value for singular values
    #OUTPUT:
    #regularized_cov:     reconstructed matrix
    #EXPLANATION: regularize a covariance matrix, by enforcing a minimum value on its eigenvalues. Explanation
    #see exercise sheet.

    values, vectors = eig(covariance)

    Q = vectors #diagonalization transformation
    L = diag(values) #diagonal matrix
    Q_inv = inv(Q) #inverse of diagonalization transformation

    L_reg = L + epsilon * L #add regularization to diagonalized covariance matrix

    regularized_cov = Q.dot(L_reg).dot(Q_inv) #undo diagonalization transformation

    return regularized_cov
