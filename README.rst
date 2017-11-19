.. image:: https://travis-ci.com/VarIr/data_mining.svg?token=Pv7ns6A7X34baaBVUTz8&branch=master
    :target: https://travis-ci.com/VarIr/data_mining

Data Mining
===========

Univie VU Data Mining course - Programming assignments

Assignment 1 - High dimensional data clustering with COPAC
----------------------------------------------------------

We implement COPAC (Correlation Partition Clustering), which

#. computes the local correlation dimensionality based on the largest eigenvalues
#. partitions the data set based on this dimension
#. calculates a Euclidean distance variant weighted with the correlation dimension, called correlation distance
#. further clusters objects within each partition with Generalized DBSCAN, requiring a minimum number of objects to be within eps range for each core point.


Installation
------------

Make sure you have a working Python3 environment (at least 3.5) with
numpy, scipy and scikit-learn packages. Consider using 
`Anaconda <https://www.anaconda.com/download/#linux>`_.
You can install COPAC from within the cloned directory with

.. code-block:: bash

  python3 setup.py install

COPAC is then available through the `cluster` package.

Example
-------

COPAC usage follows scikit-learn's cluster API.

.. code-block:: python

  from cluster import COPAC
  # load some X here ...
  copac = COPAC(k=10, mu=5, eps=.5, alpha=.85)
  y_pred = copac.fit_transform(X)

Documentation
-------------

See the PDF documentation_group08.pdf

Implementation
--------------
Published in GitHub:
https://github.com/VarIr/data_mining

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
No license so far.
