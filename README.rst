.. image:: https://travis-ci.org/VarIr/copac.svg?token=Pv7ns6A7X34baaBVUTz8&branch=master
    :target: https://travis-ci.org/VarIr/copac
.. image:: https://zenodo.org/badge/106809618.svg
   :target: https://zenodo.org/badge/latestdoi/106809618



High dimensional data clustering with COPAC
===========================================

We implement COPAC (Correlation Partition Clustering), which

#. computes the local correlation dimensionality based on the largest eigenvalues
#. partitions the data set based on this dimension
#. calculates a Euclidean distance variant weighted with the correlation dimension, called correlation distance
#. further clusters objects within each partition with Generalized DBSCAN,
   requiring a minimum number of objects to be within eps range for each core point.


Installation
------------

Clone this repository and ``pip install`` like so:

.. code-block:: bash

  git clone https://github.com/VarIr/copac.git
  cd copac
  python3 -m pip install .

COPAC is then available through the `copac` package.

Example
-------

COPAC usage follows scikit-learn's cluster API.

.. code-block:: python

  from copac import COPAC
  # load some X here ...
  copac = COPAC(k=10, mu=5, eps=.5, alpha=.85)
  y_pred = copac.fit_predict(X)


Implementation
--------------
Published in GitHub:
https://github.com/VarIr/copac

Citation
--------

The original publication of COPAC.

.. code-block:: text

    @article{Achtert2007,
             author = {Achtert, E and Bohm, C and Kriegel, H P and Kroger, P and Zimek, A},
             title = {{Robust, Complete, and Efficient Correlation Clustering}},
             journal = {Proceedings of the Seventh Siam International Conference on Data Mining},
             year = {2007},
             pages = {413--418}
    }


License
-------
This work is free open source software licensed under GPLv3.
