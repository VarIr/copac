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

import numpy as np
from scipy import sparse

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array, check_consistent_length


def copac(X, ...):
    """Perform COPAC clustering from vector array.
    Read more in the :ref:`User Guide <copac>`.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        A feature array.
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
    raise NotImplementedError
    return


class COPAC(BaseEstimator, ClusterMixin):
    """Perform COPAC clustering from vector array or distance matrix.
    Read more in the :ref:`User Guide <copac>`.
    Parameters
    ----------
    ...
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

    def __init__(self, ...):
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
