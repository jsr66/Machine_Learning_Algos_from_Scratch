import numpy as np
from numpy.linalg import eig
from numpy import diag
from numpy import dot
from numpy.linalg import inv
from numpy import array
from numpy.linalg import eig



def regularize_cov(covariance, epsilon):
    # regularize a covariance matrix, by enforcing a minimum
    # value on its singular values. Explanation see exercise sheet.
    #
    # INPUT:
    #  covariance: matrix
    #  epsilon:    minimum value for singular values
    #
    # OUTPUT:
    # regularized_cov: reconstructed matrix

    #####Insert your code here for subtask 6d#####

    values, vectors = eig(covariance)

    Q = vectors
    L = diag(values)
    Q_inv = inv(Q)

    L_reg = L + epsilon * L

    regularized_cov = Q.dot(L_reg).dot(Q_inv)

    return regularized_cov
