#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COPAC: Correlation Partition Clustering
"""

# Author: Roman Feldbauer <roman.feldbauer@ofai.at>
#         ... <>
#         ... <>
#         ... <>
#
# License: ...
from multiprocessing import cpu_count

import numpy as np
from scipy import sparse
from scipy import linalg as LA

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array, check_consistent_length
from sklearn.neighbors import NearestNeighbors


def _cdist_P(P, Q, Mhat_P):
    """ TODO write docstring 

    Parameters
    ----------
    ...
    Returns
    -------
    ...
    Notes
    -----
    The squareroot of cdist is taken later. The advantage here is to
    save some computation, as we can first take the maximum of
    two cdists, and then take the root of the 'winner' only.
    """
    PQ_diff = P - Q
    return PQ_diff @ Mhat_P @ PQ_diff.T

def _cdist(P, Q, Mhat_P, Mhat_Q):
    """ TODO write docstring

    Parameters
    ----------
    ...
    Returns
    -------
    ...
    Notes
    -----
    The sqrt is taken of the maximum distance only.
    """
    dist_PQ = _cdist_P(P, Q, Mhat_P)
    dist_QP = _cdist_P(Q, P, Mhat_Q)
    max_dist = np.max([dist_PQ, dist_QP])
    return np.sqrt(max_dist)

def copac(X, alpha=0.85, n_neighbors=5, metric='minkowski'...):
    """Perform COPAC clustering from vector array.
    Read more in the :ref:`User Guide <copac>`.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        A feature array.
    alpha : float in ]0,1[, optional, default=0.85
        Threshold of how much variance needs to be explained by Eigenvalues
    metric : str, optional, default='minkowski'
        Distance metric for nearest neighbor search
    n_neighbors : int, optional, default=5
        Size of local neighborhood for local correlation dimensionality
    n_jobs : int, optional, default=1
        Number of parallel processes. Use all cores with n_jobs=-1.
    ... : other parameters
        TBA
    Returns
    -------
    labels : array [n_samples]
        Cluster labels for each point.
    ... : possibly others
        TBA
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
    
    X = check_array(X, accept_sparse='csr')
    n, d = X.shape
    if n_jobs == -1:
        n_jobs = cpu_count()
    raise NotImplementedError

    # Calculating M^ just once requires lots of memory...
    lambda_ = np.zeros(n, dtype=int)
    M_hat = list()

    # Get nearest neighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, 
                          n_jobs=n_jobs)
    knns = nn.kneighbors(return_distances=False)
    for P, knn in enumerate(knns):
        N_P = X[knn]
        
        # Corr. cluster cov. matrix
        # TODO Is this an input parameter?
        features = ... #[0, 1, 2, ...]   # subset of dimensions
        Sigma_C = np.cov(N_P[:, features], rowvar=False, ddof=0)

        # Decompose spsd matrix, and sort Eigenvalues descending
        V_C, E_C = LA.eigh(Sigma_C)
        E_C = np.sort(E_C)[::-1]

        # Local correlation dimension
        explanation_portion = np.cumsum(E_C) / E_C.sum()
        lambda_[P] = np.searchsorted(explanation_portion, alpha, side='left')

        # Correlation distance matrix
        E_hat = (np.arange(d) > lambda_[P]).astype(int)
        M_hat.append(V_C @ np.diag(E_hat) @ V_C.T)

    # Group pts by corr. dim.
    argsorted = np.argsort(lambda_)
    edges, _ = np.histogram(lambda_[argsorted], bins=d)
    Ds = np.split(argsorted, edges)
    
    for D in Ds:
        n_D = D.shape[0]
        cdist_P = -np.ones(n_D * (n_D - 1)), dtype=np.float)
        cdist_Q = cdist_P.copy()
        ind = 0
        for i, p in enumerate(D):
            # TODO vectorize inner loop
            for j, q in enumerate(D):
                cdist_P[ind] = _cdist_P(X[p], X[q], M_hat[p])
                cdist_Q[ind] = _cdist_Q(X[q], X[p], M_hat[q])
                ind += 1
        cdist = 
    return


class COPAC(BaseEstimator, ClusterMixin):
    """Perform COPAC clustering from vector array or distance matrix.
    Read more in the :ref:`User Guide <copac>`.
    Parameters
    ----------
    ...
    alpha : float in ]0,1[, optional, default=0.85
        Threshold of how much variance needs to be explained by Eigenvalues
    n_neighbors : int, optional, default=5
        Size of local neighborhood for local correlation dimensionality
    Attributes
    ----------
    ...
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

    def __init__(self, alpha=0.85, n_neighbors=5, ...):
        self.alpha = alpha
        self.n_neighbors = n_neighbors
        #self.param0 = param0
        pass

    def fit(self, X, y=None, sample_weight=None):
        """Perform COPAC clustering from features or distance matrix.
        Parameters
        ----------
        X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
                array of shape (n_samples, n_samples)
            A feature array, or array of distances between samples if
            ``metric='precomputed'``.
        sample_weight : array, shape (n_samples,), optional
            Weight of each sample, such that a sample with a weight of at least
            ``min_samples`` is by itself a core sample; a sample with negative
            weight may inhibit its eps-neighbor from being core.
            Note that weights are absolute, and default to 1.
        y : Ignored
        """
        X = check_array(X, accept_sparse='csr')
        clust = copac(X, sample_weight=sample_weight,
                       **self.get_params())
        self.labels_ = clust
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        """Performs clustering on X and returns cluster labels.
        Parameters
        ----------
        X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
                array of shape (n_samples, n_samples)
            A feature array, or array of distances between samples if
            ``metric='precomputed'``.
        sample_weight : array, shape (n_samples,), optional
            Weight of each sample, such that a sample with a weight of at least
            ``min_samples`` is by itself a core sample; a sample with negative
            weight may inhibit its eps-neighbor from being core.
            Note that weights are absolute, and default to 1.
        y : Ignored
        Returns
        -------
        y : ndarray, shape (n_samples,)
            cluster labels
        """
        self.fit(X, sample_weight=sample_weight)
        return self.labels_


def _gdbscan(X, n_pred, min_card, w_card):
	""" TODO write docstring

	Parameters
	----------
	X : array of shape (n_samples, n_features)
		A feature array.
	n_pred : ...
		...
	min_card : ...
		...
	w_card : ...
		...

	Returns
	-------
	...
    """
    NOISE = -2
    UNCLASSIFIED = -1
    n, d = X.shape
    y = UNCLASSIFIED * np.ones(n, dtype=np.uint32)
    noise = range(n)
    cluster_id = next(noise)
    for i, object_ in enumerate(X):
		if y[i] == UNCLASSIFIED:
			if _gdbscan_expand_cluster(X, y, i, cluster_id, n_pred, 
				min_card, w_card, UNCLASSIFIED, NOISE):
				cluster_id = next(noise)
	return y

def _gdbscan_expand_cluster(X, y, i, cluster_id, n_pred, min_card, 
							w_card, UNCLASSIFIED, NOISE):
	""" TODO write (short) docstring """
	if w_card(object_) <= 0:
		y[i] = UNCLASSIFIED
		return False

	seeds = _gdbscan_neighborhood(X, i, n_pred)
	if w_card(seeds) < min_card:
		y[i] = NOISE
		return False

	y[i] = cluster_id
	seeds.delete(i)
	while len(seeds):
		j = seeds[0]
		result = _gdbscan_neighborhood(X, j, n_pred)
		if w_card(result) >= min_card:
			for P in result:
				if w_card(P) > 0 and y[P] < 0:
					if y[P] == UNCLASSIFIED:
						seeds.append(P)
					y[P] = cluster_id
		seeds.delete(j)
	return True



