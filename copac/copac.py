#!/usr/bin/env python3

"""
COPAC: Correlation Partition Clustering
"""

# Author: Roman Feldbauer <sci@feldbauer.org>
#         Elisabeth Hartel
#         Jiri Mauritz <jirmauritz at gmail dot com>
#         Thomas Turic <thomas.turic@outlook.com>

from multiprocessing import cpu_count

import numpy as np
from scipy import linalg as LA
from scipy.spatial.distance import squareform

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array


def _cdist(P, Q, Mhat_P):
    """ Correlation distance.

    Compute the corr. distance between a single sample `P` and several
    samples `Q` (vectorization for reduced runtime). The correlation
    distance is not symmetric!

    Notes
    -----
    The squareroot of cdist is taken later. The advantage here is to
    save some computation, as we can first take the maximum of
    two cdists, and then take the root of the 'winner' only.
    """
    PQ_diff = P[np.newaxis, :] - Q
    return (PQ_diff @ Mhat_P * PQ_diff).sum(axis=1)


def copac(X: np.ndarray, *,
          k: int = 10, mu: int = 5, eps: float = 0.5, alpha: float = 0.85,
          metric: str = 'euclidean', metric_params=None,
          algorithm: str = 'auto', leaf_size: int = 30, p: float = None,
          n_jobs: int = 1, sample_weight: np.ndarray=None,
          return_core_pts: bool = False):
    """Perform COPAC clustering from vector array.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        A feature array.
    k : int, optional, default=10
        Size of local neighborhood for local correlation dimensionality.
        The paper suggests k >= 3 * n_features.
    mu : int, optional, default=5
        Minimum number of points in a copac with mu <= k.
    eps : float, optional, default=0.5
        Neighborhood predicate, so that neighbors are closer than `eps`.
    alpha : float in ]0,1[, optional, default=0.85
        Threshold of how much variance needs to be explained by Eigenvalues.
        Assumed to be robust in range 0.8 <= alpha <= 0.9 [see Ref.]
    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by sklearn.metrics.pairwise.pairwise_distances
        for its metric parameter.
        If metric is "precomputed", `X` is assumed to be a distance matrix and
        must be square.
    metric_params : dict, optional
        Additional keyword arguments for the metric function.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        The algorithm to be used by the scikit-learn NearestNeighbors module
        to compute pointwise distances and find nearest neighbors.
        See NearestNeighbors module documentation for details.
    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree or cKDTree. This can affect the speed
        of the construction and query, as well as the memory required
        to store the tree. The optimal value depends
        on the nature of the problem.
    p : float, optional
        The power of the Minkowski metric to be used to calculate distance
        between points.
    n_jobs : int, optional, default=1
        Number of parallel processes. Use all cores with n_jobs=-1.
    sample_weight : None
        Sample weights
    return_core_pts : bool
        Return clusters labels and core point indices for each correlation dimension.

    Returns
    -------
    labels : array [n_samples]
        Cluster labels for each point. Noisy samples are given the label -1.
    core_pts_ind : dict[int, array]
        Indices of core points for each correlation dimension (only if ``return_core_pts=True``).

    References
    ----------
    Elke Achtert, Christian Bohm, Hans-Peter Kriegel, Peer Kroger,
    A. Z. (n.d.). Robust, complete, and efficient correlation
    clustering. In Proceedings of the Seventh SIAM International
    Conference on Data Mining, April 26-28, 2007, Minneapolis,
    Minnesota, USA (2007), pp. 413–418.
    """
    n, d = X.shape
    data_dtype = X.dtype
    y = -np.ones(n, dtype=int)
    if n_jobs == -1:
        n_jobs = cpu_count()

    # Calculating M^ just once requires more memory, but saves computation
    lambda_ = np.zeros(n, dtype=int)
    M_hat = list()

    # Get nearest neighbors
    nn = NearestNeighbors(n_neighbors=k, metric=metric, algorithm=algorithm,
                          leaf_size=leaf_size, metric_params=metric_params,
                          p=p, n_jobs=n_jobs)
    nn.fit(X)
    knns = nn.kneighbors(return_distance=False)
    for P, knn in enumerate(knns):
        N_P = X[knn]

        # Correlation copac covariance matrix
        Sigma = np.cov(N_P[:, :], rowvar=False, ddof=0)

        # Decompose spsd matrix, and sort Eigenvalues descending
        E, V = LA.eigh(Sigma)
        E = np.sort(E)[::-1]

        # Local correlation dimension
        explanation_portion = np.cumsum(E) / E.sum()
        lambda_P = np.searchsorted(explanation_portion, alpha, side='left')
        lambda_P += 1
        lambda_[P] = lambda_P
        # Correlation distance matrix
        E_hat = (np.arange(1, d + 1) > lambda_P).astype(int)
        M_hat.append(V @ np.diag(E_hat) @ V.T)

    # Group points by corr. dim.
    argsorted = np.argsort(lambda_)
    edges, _ = np.histogram(lambda_[argsorted], bins=np.arange(1, d + 2))
    Ds = np.split(argsorted, np.cumsum(edges))
    # Loop over partitions according to local corr. dim.
    max_label = 0
    used_y = np.zeros_like(y, dtype=int)
    core_pts = {}
    for dim, D in enumerate(Ds, start=1):
        n_D = D.shape[0]
        cdist_P = -np.ones(n_D * (n_D - 1) // 2, dtype=data_dtype)
        cdist_Q = -np.ones((n_D, n_D), dtype=data_dtype)
        start = 0
        # Calculate triu part of distance matrix
        for i in range(0, n_D - 1):
            p = D[i]
            # Vectorized inner loop
            q = D[i + 1:n_D]
            stop = start + n_D - i - 1
            cdist_P[start:stop] = _cdist(X[p], X[q], M_hat[p])
            start = stop
        # Calculate tril part of distance matrix
        for i in range(1, n_D):
            q = D[i]
            p = D[0:i]
            cdist_Q[i, :i] = _cdist(X[q], X[p], M_hat[q])
        # Extract tril to 1D array
        # TODO simplify...
        cdist_Q = cdist_Q.T[np.triu_indices_from(cdist_Q, k=1)]
        cdist = np.block([[cdist_P], [cdist_Q]])
        # Square root of the higher value of cdist_P, cdist_Q
        cdist = np.sqrt(cdist.max(axis=0))

        # Perform DBSCAN with full distance matrix
        cdist = squareform(cdist)
        dbscan = DBSCAN(eps=eps, min_samples=mu, metric="precomputed", n_jobs=n_jobs)
        labels = dbscan.fit_predict(X=cdist, sample_weight=sample_weight)
        core_pts[dim] = dbscan.core_sample_indices_

        # Each DBSCAN run is unaware of previous ones,
        # so we need to keep track of previous copac IDs
        y_D = labels + max_label
        new_labels = np.unique(labels[labels >= 0]).size
        max_label += new_labels
        # Set copac labels in `y`
        y[D] = y_D
        used_y[D] += 1
    assert np.all(used_y == 1), "Not all samples were handled exactly once!"

    if return_core_pts:
        return y, core_pts
    else:
        return y


class COPAC(BaseEstimator, ClusterMixin):
    """Perform COPAC clustering from vector array.

    Parameters
    ----------
    k : int, optional, default=10
        Size of local neighborhood for local correlation dimensionality.
        The paper suggests k >= 3 * n_features.
    mu : int, optional, default=5
        Minimum number of points in a copac with mu <= k.
    eps : float, optional, default=0.5
        Neighborhood predicate, so that neighbors are closer than `eps`.
    alpha : float in ]0,1[, optional, default=0.85
        Threshold of how much variance needs to be explained by Eigenvalues.
        Assumed to be robust in range 0.8 <= alpha <= 0.9 [see Ref.]
    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by sklearn.metrics.pairwise.pairwise_distances
        for its metric parameter.
        If metric is "precomputed", `X` is assumed to be a distance matrix and
        must be square.
    metric_params : dict, optional
        Additional keyword arguments for the metric function.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        The algorithm to be used by the scikit-learn NearestNeighbors module
        to compute pointwise distances and find nearest neighbors.
        See NearestNeighbors module documentation for details.
    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree or cKDTree. This can affect the speed
        of the construction and query, as well as the memory required
        to store the tree. The optimal value depends
        on the nature of the problem.
    p : float, optional
        The power of the Minkowski metric to be used to calculate distance
        between points.
    n_jobs : int, optional, default=1
        Number of parallel processes. Use all cores with n_jobs=-1.

    Attributes
    ----------
    labels_ : array, shape = [n_samples]
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.

    Notes
    -----
    ...
    References
    ----------
    Elke Achtert, Christian Bohm, Hans-Peter Kriegel, Peer Kroger,
    A. Z. (n.d.). Robust, complete, and efficient correlation
    clustering. In Proceedings of the Seventh SIAM International
    Conference on Data Mining, April 26-28, 2007, Minneapolis,
    Minnesota, USA (2007), pp. 413–418.
    """

    def __init__(self, k=10, mu=5, eps=0.5, alpha=0.85,
                 metric='euclidean', metric_params=None, algorithm='auto',
                 leaf_size=30, p=None, n_jobs=1):
        self.k = k
        self.mu = mu
        self.eps = eps
        self.alpha = alpha
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs

    def fit(self, X, y=None, sample_weight=None, return_core_pts=False):
        """Perform COPAC clustering from features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            A feature array.
        sample_weight : array, shape (n_samples,), optional
            Weight of each sample, such that a sample with a weight of at least
            ``min_samples`` is by itself a core sample; a sample with negative
            weight may inhibit its eps-neighbor from being core.
            Note that weights are absolute, and default to 1.
            CURRENTLY IGNORED.
        y : Ignored
        return_core_pts : bool
            Return cluster labels and core points per correlation dimension
        """
        X: np.ndarray = check_array(X)
        result = copac(
            X=X,
            sample_weight=sample_weight,
            return_core_pts=return_core_pts,
            **self.get_params(),
        )
        if return_core_pts:
            clust, core_pts = result
            self.core_point_indices_ = core_pts
        else:
            clust = result
        self.labels_ = clust
        return self

    def fit_predict(self, X, y=None, sample_weight=None, return_core_pts=False):
        """Performs clustering on X and returns copac labels.

        Parameters
        ----------
        X : ndarray matrix of shape (n_samples, n_features)
            A feature array.
        sample_weight : array, shape (n_samples,), optional
            Weight of each sample, such that a sample with a weight of at least
            ``min_samples`` is by itself a core sample; a sample with negative
            weight may inhibit its eps-neighbor from being core.
            Note that weights are absolute, and default to 1.
            CURRENTLY IGNORED.
        y : Ignored
        return_core_pts : bool
            Return cluster labels and core points per correlation dimension

        Returns
        -------
        y : ndarray, shape (n_samples,)
            copac labels
        """
        self.fit(X, sample_weight=sample_weight, return_core_pts=return_core_pts)
        if return_core_pts:
            return self.labels_, self.core_point_indices_
        else:
            return self.labels_
