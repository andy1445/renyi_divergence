def renyi_alpha_divergence(x, y, k=9, alpha=0.8):
    """
    Non-parametric estimator of Renyi alpha divergence proposed in
    http://proceedings.mlr.press/v15/poczos11a/poczos11a.pdf
    Inputs:
    x,y:  ndarray of shape n,1 (It can be extended for more dimensions, but first double check paper)
       alpha: scalar
        k: number of neighbors
    Output:
        Renyi-alpha divergence
    """
    knn_x = NearestNeighbors(n_neighbors=k+1)
    knn_x.fit(x)
    knn_y = NearestNeighbors(n_neighbors=k)
    knn_y.fit(y)
    n = len(x)
    m = len(y)
    renyi_divergence = 0
    for i in range(len(x)):
        pk = knn_x.kneighbors(x[i].reshape(-1, x.shape[1]), return_distance=True)[0][0][k] # has itself in position 0
        vk = knn_y.kneighbors(x[i].reshape(-1, x.shape[1]), return_distance=True)[0][0][k-1] # k-1 because it does not have itself in position 0
        pk = pk ** x.shape[1]
        vk = vk ** x.shape[1]
#         value = ((n-1)*pk)/(m*vk + 1e-10)
        value = ((n-1)*pk)/(m*vk) if m*vk != 0 else 0
        renyi_divergence += value**(1-alpha)
    bias = gamma(k)**2/(gamma(k-alpha+1)*gamma(k+alpha-1))
    renyi_divergence = renyi_divergence*bias/n
    renyi_divergence = np.log(renyi_divergence)/(alpha-1)
    return renyi_divergence
