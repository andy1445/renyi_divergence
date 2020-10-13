import numpy as np
import scipy.spatial
from scipy.special import gamma

def renyi_alpha_divergence(X, Y, k=3, alpha=0.8):
    n = len(X)
    m = len(Y)
    distance_X  = scipy.spatial.distance.cdist(X, X) # n by n distance matrix (euclidean)
    distance_XY = scipy.spatial.distance.cdist(X, Y) # n by m distance matrix (euclidean)
    # the Euclidean distance of the kth nearest neighbor of X_i in the sample X_1:n
    rho_k = np.sort(distance_X, axis=1)[:, k]
    # the Euclidean distance of the kth nearest neighbor of X_i in the sample Y_1:m
    nu_k  = np.sort(distance_XY, axis=1)[:, k]
    B_k_alpha = gamma(k)**2/(gamma(k-alpha+1) * gamma(k+alpha-1))
    D_hat = np.divide(rho_k, nu_k+1e-6) * (n-1) / m
    D_hat = np.power(D_hat, 1-alpha)
    D_hat = 1/n * np.sum(D_hat)
    D_hat = D_hat * B_k_alpha
    return D_hat
