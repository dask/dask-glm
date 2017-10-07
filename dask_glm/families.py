from __future__ import absolute_import, division, print_function

from dask_glm.utils import dot, exp, log1p, sigmoid, RegistryClass


class Family(RegistryClass):
    """Base class methods for distribution.

    This class represents the required methods to add a new distribution to work
    with the algorithms.
    """

    def loglikelihood(self, Xbeta, y):
        raise NotImplementedError

    def pointwise_loss(self, beta, X, y):
        """Loss, evaluated point-wise."""
        beta, y = beta.ravel(), y.ravel()
        Xbeta = X.dot(beta)
        return self.loglikelihood(Xbeta, y)

    def pointwise_gradient(self, beta, X, y):
        """Loss, evaluated point-wise."""
        beta, y = beta.ravel(), y.ravel()
        Xbeta = X.dot(beta)
        return self.gradient(Xbeta, X, y)

    def gradient(self, Xbeta, x, y):
        raise NotImplementedError

    def hessian(self, Xbeta, x):
        raise NotImplementedError


class Logistic(Family):
    """Implements methods for `Logistic regression`_,
    useful for classifying binary outcomes.

    .. _Logistic regression: https://en.wikipedia.org/wiki/Logistic_regression
    """
    name = 'logistic'

    def loglikelihood(self, Xbeta, y):
        """
        Evaluate the logistic loglikelihood

        Parameters
        ----------
        Xbeta : array, shape (n_samples, n_features)
        y : array, shape (n_samples)
        """
        enXbeta = exp(-Xbeta)
        return (Xbeta + log1p(enXbeta)).sum() - dot(y, Xbeta)

    def gradient(self, Xbeta, X, y):
        """Logistic gradient"""
        p = sigmoid(Xbeta)
        return dot(X.T, p - y)

    def hessian(self, Xbeta, X):
        """Logistic hessian"""
        p = sigmoid(Xbeta)
        return dot(p * (1 - p) * X.T, X)


class Normal(Family):
    """Implements methods for `Linear regression`_,
    useful for modeling continuous outcomes.

    .. _Linear regression: https://en.wikipedia.org/wiki/Linear_regression
    """
    name = 'normal'

    def loglikelihood(self, Xbeta, y):
        return ((y - Xbeta) ** 2).sum()

    def gradient(self, Xbeta, X, y):
        return 2 * dot(X.T, Xbeta) - 2 * dot(X.T, y)

    def hessian(self, Xbeta, X):
        return 2 * dot(X.T, X)


class Poisson(Family):
    """
    This implements `Poisson regression`_, useful for
    modelling count data.


    .. _Poisson regression: https://en.wikipedia.org/wiki/Poisson_regression
    """
    name = 'poisson'

    def loglikelihood(self, Xbeta, y):
        eXbeta = exp(Xbeta)
        yXbeta = y * Xbeta
        return (eXbeta - yXbeta).sum()

    def gradient(self, Xbeta, X, y):
        eXbeta = exp(Xbeta)
        return dot(X.T, eXbeta - y)

    def hessian(self, Xbeta, X):
        eXbeta = exp(Xbeta)
        x_diag_eXbeta = eXbeta[:, None] * X
        return dot(X.T, x_diag_eXbeta)
