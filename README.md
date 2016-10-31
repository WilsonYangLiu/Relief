[![Build Status](https://travis-ci.org/rhiever/ReliefF.svg?branch=master)](https://travis-ci.org/rhiever/ReliefF)
[![Code Health](https://landscape.io/github/rhiever/ReliefF/master/landscape.svg?style=flat)](https://landscape.io/github/rhiever/ReliefF/master)
[![Coverage Status](https://coveralls.io/repos/github/rhiever/ReliefF/badge.svg?branch=master)](https://coveralls.io/github/rhiever/ReliefF?branch=master)
![Python 2.7](https://img.shields.io/badge/python-2.7-blue.svg)
![Python 3.5](https://img.shields.io/badge/python-3.5-blue.svg)
![License](https://img.shields.io/badge/license-MIT%20License-blue.svg)
[![PyPI version](https://badge.fury.io/py/ReliefF.svg)](https://badge.fury.io/py/ReliefF)

# ReliefF

[![Join the chat at https://gitter.im/rhiever/ReliefF](https://badges.gitter.im/rhiever/ReliefF.svg)](https://gitter.im/rhiever/ReliefF?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

This package contains implementations of the [ReliefF](https://en.wikipedia.org/wiki/Relief_(feature_selection)) family of feature selection algorithms. **It is still under active development** and we encourage you to check back on this repository regularly for updates.

These algorithms excel at identifying features that are predictive of the outcome in supervised learning problems, and are especially good at identifying feature interactions that are normally overlooked by standard feature selection algorithms.

The main benefit of ReliefF algorithms is that they identify feature interactions without having to exhaustively check every pairwise interaction, thus taking significantly less time than exhaustive pairwise search.

ReliefF algorithms are commonly applied to genetic analyses, where epistasis (i.e., feature interactions) is common. However, the algorithms implemented in this package can be applied to any supervised classification data set.

However, note that this implementation of ReliefF **currently only works with categorical features**. We are working on expanding the algorithm to support continuous features as well.

## License

Please see the [repository license](https://github.com/rhiever/ReliefF/blob/master/LICENSE) for the licensing and usage information for the ReliefF package.

Generally, we have licensed the ReliefF package to make it as widely usable as possible.

## Installation

ReliefF is built on top of the following existing Python packages:

* NumPy

* SciPy

* scikit-learn

All of the necessary Python packages can be installed via the [Anaconda Python distribution](https://www.continuum.io/downloads), which we strongly recommend that you use. We also strongly recommend that you use Python 3 over Python 2 if you're given the choice.

NumPy, SciPy, and scikit-learn can be installed in Anaconda via the command:

```
conda install numpy scipy scikit-learn
```

Once the prerequisites are installed, you should be able to install ReliefF with a pip command:

```
pip install relieff
```

Please [file a new issue](https://github.com/rhiever/ReliefF/issues/new) if you run into installation problems.

## Example

ReliefF has been coded with a scikit-learn-like interface to be easy to use. The typical `fit`, `transform`, and `fit_transform` methods are available for every algorithm.

```python
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from ReliefF import ReliefF

digits = load_digits(2)
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

fs = ReliefF(n_neighbors=100, n_features_to_keep=5)
X_train = fs.fit_transform(X_train, y_train)
X_test_subset = fs.transform(X_test)
print(X_test.shape, X_test_subset.shape)
```

This code should output `(90, 64) (90, 5)`, indicating that ReliefF successfully subset the features down to the 5 most predictive features.

## Contributing to ReliefF

We welcome you to [check the existing issues](https://github.com/rhiever/ReliefF/issues/) for bugs or enhancements to work on. If you have an idea for an extension to the ReliefF package, please [file a new issue](https://github.com/rhiever/ReliefF/issues/new) so we can discuss it.

## Having problems or have questions about ReliefF?

Please [check the existing open and closed issues](https://github.com/rhiever/ReliefF/issues?utf8=%E2%9C%93&q=is%3Aissue) to see if your issue has already been attended to. If it hasn't, [file a new issue](https://github.com/rhiever/ReliefF/issues/new) on this repository so we can review your issue.

## Citing ReliefF

If you use this software in a publication, please consider citing it. You can cite the repository directly with the following DOI:

[![DOI](https://zenodo.org/badge/20747/rhiever/ReliefF.svg)](https://zenodo.org/badge/latestdoi/20747/rhiever/ReliefF)

## Support for ReliefF

The ReliefF package was developed in the [Computational Genetics Lab](http://epistasis.org) with funding from the [NIH](http://www.nih.gov). We're incredibly grateful for their support during the development of this project.
