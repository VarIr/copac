"""
Testing for Clustering methods
"""
import unittest

import numpy as np

from sklearn.metrics.cluster import v_measure_score
from sklearn.utils.testing import assert_equal, assert_array_equal
from sklearn.datasets.samples_generator import make_blobs

from ..copac import COPAC


class TestCopac(unittest.TestCase):

    def setUp(self):
        """ Set up very simple data set """
        self.n_clusters = 2
        self.centers = np.array([[3, 3], [-3, -3]]) + 10
        self.X, self.y = make_blobs(n_samples=60, n_features=2,
                                    centers=self.centers, cluster_std=0.4,
                                    shuffle=True, random_state=0)
        self.v = v_measure_score(self.y, self.y)

    def tearDown(self):
        del self.n_clusters, self.centers, self.X

    def test_copac(self):
        """ Minimal test that COPAC runs at all. """
        k = 40
        mu = 10
        eps = 2
        alpha = 0.85
        copac = COPAC(k=k, mu=mu, eps=eps, alpha=alpha)
        y_pred = copac.fit_predict(self.X)
        v = v_measure_score(self.y, y_pred)
        # Must score perfectly on very simple data
        assert_equal(self.v, v)
        # Check correct labels_ attribute
        copac = COPAC(k=k, mu=mu, eps=eps, alpha=alpha)
        copac.fit(self.X)
        assert_array_equal(copac.labels_, y_pred)

if __name__ == "__main__":
    unittest.main()
