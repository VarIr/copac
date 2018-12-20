#!/usr/bin/env python3

"""
Installation:
-------------
In the console change to the folder containing this file.
To build the package copac:
python3 setup.py build
To install the package:
python3 setup.py install
To test the installation:
python3 setup.py test
If this succeeds with an 'OK' message, you are ready to go.
"""
import sys
if sys.version_info < (3, 6):
    sys.stdout.write("COPAC requires Python 3.6\n"
                     "Please try to run as python3 setup.py or\n"
                     "update your Python environment.\n"
                     "Consider using Anaconda for easy package handling.\n")
    sys.exit(1)

try:
    import numpy
    import scipy
    import sklearn
except ImportError:
    sys.stdout.write("COPAC requires numpy, scipy and scikit-learn. "
                     "Please make sure these packages are available locally. "
                     "Consider using Anaconda for easy package handling.\n")

setup_options = {}

try:
    from setuptools import setup
    setup_options['test_suite'] = 'copac/tests/test_copac.py'
except ImportError:
    from distutils.core import setup
    import warnings
    warnings.warn("setuptools not found, resorting to distutils. "
                  "Unit tests won't be discovered automatically.")

setup(
    name="copac",
    version="0.1",
    author="Roman Feldbauer",
    author_email="roman.feldbauer@ofai.at",
    maintainer="Roman Feldbauer",
    maintainer_email="roman.feldbauer@ofai.at",
    description="Correlation clustering",
    license="",
    keywords=["machine learning", "data science", "data mining"],
    url="https://github.com/VarIr/copac",
    # packages=['copac'],
    # package_data={'...': ['example_datasets/*']},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering"
    ],
    **setup_options
)
