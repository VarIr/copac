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

    lambda_ = np.zeros(n, dtype=int)
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
    # Group pts by corr. dim.
    argsorted = np.argsort(lambda_)
    edges, _ = np.histogram(lambda_[argsorted], bins=d)
    D = np.split(argsorted, edges)
    
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
