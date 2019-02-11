import numpy as np
from estGaussMixEM import estGaussMixEM
from getLogLikelihood import getLogLikelihood, Gaussian


def skinDetection(ndata, sdata, K, n_iter, epsilon, theta, img):
    # Skin Color detector
    #
    # INPUT:
    # ndata         : data for non-skin color
    # sdata         : data for skin-color
    # K             : number of modes
    # n_iter        : number of iterations
    # epsilon       : regularization parameter
    # theta         : threshold
    # img           : input image
    #
    # OUTPUT:
    # result        : Result of the detector for every image pixel

    #####Insert your code here for subtask 1g#####

    #Learn densities for skin and non-skin
    weights_n, means_n, covariances_n = estGaussMixEM(ndata, K, n_iter, epsilon)
    weights_s, means_s, covariances_s = estGaussMixEM(sdata, K, n_iter, epsilon)
    result = np.zeros((img.shape[0], img.shape[1]))
    #print("result.shape: " + str(result.shape))
    theta = 1

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            p_cs = 0
            p_cn = 0
            for k in range(K):
                p_cs += weights_s[k] * Gaussian(img[i][j], means_s[k,:], covariances_s[:,:,k])
                p_cn += weights_n[k] * Gaussian(img[i][j], means_n[k,:], covariances_n[:,:,k])
            if p_cs / p_cn > theta:
                result[i,j] = 1
    return result
