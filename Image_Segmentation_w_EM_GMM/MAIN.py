import numpy as np
from numpy import diag
from numpy import dot
from numpy.linalg import inv
from numpy import array
from numpy.linalg import eig
import matplotlib.pyplot as plt
from scipy import misc
import imageio
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
#from imageio import imread


def Gaussian(x, mu, Sigma):
    #INPUT
    #mu: mean, 1xD array
    #Sigma: covariance matrix, DxD array
    #x: position, 1xD array
    #OUTPUT
    #y: Value of Gaussian output
    D = len(mu)
    y = (1./(((2*np.pi)**(D/2))*np.sqrt(np.linalg.det(Sigma)))) * np.exp(-.5*np.dot(x - mu, np.dot(np.linalg.inv(Sigma), x - mu)))
    return y

def getLogLikelihood(means, weights, covariances, X): # Log Likelihood estimation
    #INPUT
    #means: Mean for each Gaussian KxD
    #weights: Weight vector 1xK for K Gaussians
    #covariances: Covariance matrices for each gaussian DxDxK
    #X: Input data NxD
    #N is number of data points
    #D is the dimension of the data points
    #K is number of gaussians
    #OUTPUT
    #logLikelihood: log-likelihood
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

def EStep(means, covariances, weights, X):
    #INPUT
    #means: Mean for each Gaussian KxD
    #covariances: Covariance matrices for each Gaussian DxDxK
    #weights: Weight vector 1xK for K Gaussians
    #X: Input data NxD
    #OUTPUT
    #logLikelihood: Log-likelihood (a scalar).
    #gamma: NxK matrix of responsibilities for N datapoints and K Gaussians.
    #EXPLANATION: Expectation step of the EM Algorithm. N is number of data points, D is the dimension of the data
    #points, K is number of Gaussians.
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

def MStep(gamma, X):
    #INPUT
    #gamma: NxK matrix of responsibilities for N datapoints and K Gaussians.
    #X: Input data (NxD matrix for N datapoints of dimension D).
    #OUTPUT
    #logLikelihood: Log-likelihood (a scalar).
    #means: Mean for each gaussian (KxD).
    #weights: Vector of weights of each gaussian (1xK).
    #covariances: Covariance matrices for each component(DxDxK).
    #EXPLANATION
    #Maximization step of the EM Algorithm. N is number of data points, D the dimension of the data
    #points, K the number of Gaussians.

    N = gamma.shape[0]  # number of examples
    K = gamma.shape[1]  # number of classes
    D = X.shape[1]  # number of features

    N_class = np.zeros((K))  # average number per class, initialize to zeros.

    weights = np.zeros((K))  # weights for different classes. initialize to zeros.
    means = np.zeros((K, D))  # means for different classes. initialize to zeros.
    covariances = np.zeros((D, D, K))  # covariances for different classes, initialize to zeros.

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
            covariances[:, :, k] += gamma[n][k] * np.outer((x_n - mu_k), (x_n - mu_k))
        covariances[:, :, k] = covariances[:, :, k] / N_k

    logLikelihood = getLogLikelihood(means, weights, covariances, X)

    return weights, means, covariances, logLikelihood

def regularize_cov(covariance, epsilon):
    #INPUT
    #covariance: DxD matrix
    #epsilon: regulator, minimum value for singular values
    #OUTPUT
    #regularized_cov: reconstructed matrix
    #EXPLANATION
    #regularize a covariance matrix, by enforcing a minimum value on its eigenvalues. Explanation
    #see exercise sheet.

    values, vectors = eig(covariance)

    Q = vectors #diagonalization transformation
    L = diag(values) #diagonal matrix
    Q_inv = inv(Q) #inverse of diagonalization transformation

    L_reg = L + epsilon * L #add regularization to diagonalized covariance matrix

    regularized_cov = Q.dot(L_reg).dot(Q_inv) #undo diagonalization transformation

    return regularized_cov

def estGaussMixEM(data, K, n_iters, epsilon):
    #INPUT
    #data: input data, N observations, D dimensional
    #K: number of mixture components (modes)
    #n_iters: number of iterations of EM
    #OUTPUT
    #weights: mixture weights
    #means: means of gaussians
    #covariances: covariance matrices of gaussians
    #EXPLANATION
    #EM algorithm for estimation gaussian mixture mode

    D = data.shape[1]

    #Initialize weights
    weights = np.ones(K) / K
    covariances = np.zeros((D, D, K))

    #Initialize means and covariances with Kmeans
    kmeans = KMeans(n_clusters=K, n_init=10).fit(data)
    cluster_idx = kmeans.labels_
    means = kmeans.cluster_centers_
    #Create initial covariance matrices
    for j in range(K):
        data_cluster = data[cluster_idx == j]
        min_dist = np.inf
        for i in range(K):
            # compute sum of squared distances in cluster
            dist = np.mean(euclidean_distances(data_cluster, [means[i]], squared=True))
            if dist < min_dist:
                min_dist = dist
        covariances[:, :, j] = np.eye(D) * min_dist

    #Loop through n_iters iterations of E-step, then M-step
    for i in range(n_iters):
        logLikelihood, gamma = EStep(means, covariances, weights, data)
        weights, means, covariances, logLikelihood = MStep(gamma, data)

    #Regularize covariance matrices
    for k in range(K):
        covariances[:, :, k] = regularize_cov(covariances[:,:,k], epsilon)

    return [weights, means, covariances]

def plotGaussian(mu, sigma):
    dimension = mu.shape[0]
    if len(mu.shape) > 1:
        n_components = mu.shape[1]
    else:
        n_components = 1
    plt.subplot()
    if dimension == 2:
        if n_components == 1 and sigma.shape == (2, 2):
            n = 36
            phi = np.arange(0, n, 1) / (n-1) * 2 * np.pi
            epoints = np.multiply(np.sign(sigma), np.sqrt(np.abs(sigma))).dot([np.cos(phi), np.sin(phi)]) + mu[:, np.newaxis]
            plt.plot(epoints[0, :], epoints[1, :], 'r')
        else:
            print('ERROR: size mismatch in mu or sigma\n')
    else:
        raise ValueError('Only dimension 2 is implemented.')

def plotModes(means, covMats, X):
    print("In plotModes, means.shape: " + str(means.shape))
    print("In plotModes, covariances.shape: " + str(covMats.shape))
    plt.subplot()
    plt.scatter(X[:, 0], X[:, 1])
    M = means.shape[1]

    for i in range(M):
        plotGaussian(means[:, i], covMats[:, :, i])

def skinDetection(ndata, sdata, K, n_iter, epsilon, theta, img):
    #INPUT
    #ndata: data for non-skin color
    #sdata: ata for skin-color
    #K: number of modes
    #n_iter: number of iterations
    #epsilon: regularization parameter
    #theta: threshold
    #img: input image
    #OUTPUT
    #result: Result of the detector for every image pixel
    #EXPLANATION: Skin Color detector

    #Learn densities for skin and non-skin
    weights_n, means_n, covariances_n = estGaussMixEM(ndata, K, n_iter, epsilon)
    weights_s, means_s, covariances_s = estGaussMixEM(sdata, K, n_iter, epsilon)
    result = np.zeros((img.shape[0], img.shape[1]))

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

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out




if __name__ == '__main__':
    '''
    #SIMULATED DATA GENERATION
    #Generate three simulated data sets and save them to file. Learn Gaussian mixture model. Learned parameters should be close to ones that defined data set.
    #Define weights, means, and covariance matrices of mixtures used to generate the three data sets.Then learn GMM with EM on examples of
    #skin vs non-skin pixels; apply learned GMM to perform image segmentation - specifically, a 'skin detector' - on a real photo.
    w1 = [1. / 3., 1. / 3., 1. / 3.]
    means1 = [[-4, 4], [0, 0], [4, 4]]
    covariances1 = .1 * np.array([[[1, 0], [0, 1]], [[2, 0], [0, 2]], [[3, 0], [0, 3]]])
    np.swapaxes(covariances1, 0, 2) #switch class index from last to first to work with above functions

    w2 = [.7, .1, .1, .1]
    means2 = [[-4, 0], [0, 0], [4, 0], [8, 0]]
    covariances2 = .1 * np.array([[[2, -1],[-1, 3]], [[1, -2],[-2, 1]], [[1, 0], [0, 2]], [[3, 0], [0, 1]]])
    np.swapaxes(covariances2, 0, 2) #switch class index from last to first to work with above functions

    w3 = [.6, .2, .1, .05, .05]
    means3 = [[-2, -2], [-2, 2], [2, -2], [2, 2], [0, 0]]
    covariances3 = .1 * np.array([[[2, -1],[-1, 2]], [[1, -2], [-2, 1]], [[1, 0], [0, 2]], [[2, 0], [0, 2]], [[3, 0], [0, 1]]])
    np.swapaxes(covariances3, 0, 2) #switch class index from last to first to work with above functions

    #Collect Mixture parameters for different datasets
    params = [[w1, means1, covariances1], [w2, means2, covariances2], [w3, means3, covariances3]]

    #Set number of data points N in each set
    N = 500

    #Sample data sets
    for i in range(3):
        print('Generating dataset' + str(i+1))
        f = open('Data' + str(i + 1) + '.txt', 'w+')
        for j in range(N):
            weights = params[i][0]
            K = len(weights)
            means = params[i][1]
            covariances = params[i][2]

            cluster_num = np.random.choice(np.arange(K), p=weights)

            mu = means[cluster_num]
            sigma = covariances[cluster_num]

            x_sample = np.random.multivariate_normal(mu, sigma)

            f.write(str(x_sample[0]) + ' ' + str(x_sample[1]) + '\n')
    '''

    #load simulated datasets
    data = [[], [], []]
    data[0] = np.loadtxt('Data1.txt')
    data[1] = np.loadtxt('Data2.txt')
    data[2] = np.loadtxt('Data3.txt')

    #parameters for training
    epsilon = 0.0001  #regularization
    K_vals = [3, 4, 5]  #lists actual number of clusters in each dataset  #number of desired clusters for each dataset
    n_iter = 10  #number of iterations
    skin_n_iter = 5
    skin_epsilon = 0.0001
    skin_K = 1
    theta = 2.0  #threshold for skin detection

    #PLOT ALL 3 DATASETS. COMPUTE GMM W APPROPRIATE NUMBER OF CLUSTERS ON ALL 3 DATASETS
    print('\n')
    print('evaluating EM for GMM on all datasets')

    for i in range(3):
        print('evaluating on dataset {0}\n'.format(i + 1))
        #compute GMM
        weights, means, covariances = estGaussMixEM(data[i], K_vals[i], n_iter, epsilon)
        print('shape of learned covariance matrices: ' + str(covariances.shape))
        #plot result
        plt.subplot()
        plotModes(np.transpose(means), covariances, data[i])
        plt.title('Data {0}'.format(i + 1))
        plt.show()

    #INFER CLUSTER NUMBER FROM LOG LIKELIHOOD PLOT
    #infer number of clusters from data by examining plot of log likelihood with number of clusters. look for 'elbow'.
    num = 14
    logLikelihood = np.zeros(num)

    for i in range(3):
        logLikelihood = np.zeros(num)
        print('')
        print('')
        print('i: ' + str(i))
        for k in range(num):
            # compute GMM
            K = k + 1
            weights, means, covariances = estGaussMixEM(data[i], K, n_iter, epsilon)
            logLikelihood[k] = getLogLikelihood(means, weights, covariances, data[i])
        print('weights: ' + str(weights))
        print('means: ' + str(means))
        print('covariances: ' + str(covariances))

        # plot result
        plt.subplot()
        plt.plot(range(1, num+1), logLikelihood)
        plt.title('Loglikelihood for different number of k on Data ' + str(i+1))
        plt.show()

    #SKIN DETECTOR
    sdata = np.loadtxt('skin.dat')
    ndata = np.loadtxt('non-skin.dat')

    img = im2double(imageio.imread('IMG_3064.png'))
    #img = im2double(misc.imread('faces.png'))
    print("img.shape: " + str(img.shape))

    skin = skinDetection(ndata, sdata, skin_K, skin_n_iter, skin_epsilon, theta, img)
    plt.imshow(skin)
    plt.show()
    misc.imsave('skin_detection.png', skin)



