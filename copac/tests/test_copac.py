"""
Testing for Clustering methods
"""
import pytest

import numpy as np
from sklearn.metrics.cluster import v_measure_score
from sklearn.datasets import make_blobs

from ..copac import COPAC


@pytest.mark.parametrize("return_core_pts", [True, False])
def test_copac(return_core_pts):
    """ Minimal test that COPAC runs at all. """
    # Set up very simple data set
    n_clusters = 2
    centers = np.array([[3, 3], [-3, -3]]) + 10
    X, y = make_blobs(n_samples=60, n_features=2,
                      centers=centers, cluster_std=0.4,
                      shuffle=True, random_state=0)
    v_true = v_measure_score(y, y)

    k = 40
    mu = 10
    eps = 2
    alpha = 0.85

    copac = COPAC(k=k, mu=mu, eps=eps, alpha=alpha)
    y_pred = copac.fit_predict(X, return_core_pts=return_core_pts)
    if return_core_pts:
        y_pred, core_pts_ind = y_pred
        assert isinstance(core_pts_ind, dict)
    v_pred = v_measure_score(y, y_pred)
    # Must score perfectly on very simple data
    np.testing.assert_equal(v_true, v_pred)
    # Check correct labels_ attribute
    copac = COPAC(k=k, mu=mu, eps=eps, alpha=alpha)
    copac.fit(X)
    np.testing.assert_array_equal(copac.labels_, y_pred)
