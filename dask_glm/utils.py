from __future__ import absolute_import, division, print_function

import inspect
import sys

import dask.array as da
import numpy as np
from functools import wraps
from multipledispatch import dispatch


def normalize(algo):
    @wraps(algo)
    def normalize_inputs(X, y, *args, **kwargs):
        normalize = kwargs.pop('normalize', True)
        if normalize:
            mean, std = da.compute(X.mean(axis=0), X.std(axis=0))
            mean, std = mean.copy(), std.copy()  # in case they are read-only
            intercept_idx = np.where(std == 0)
            if len(intercept_idx[0]) > 1:
                raise ValueError('Multiple constant columns detected!')
            mean[intercept_idx] = 0
            std[intercept_idx] = 1
            mean = mean if len(intercept_idx[0]) else np.zeros_like(mean)
            Xn = (X - mean) / std
            out = algo(Xn, y, *args, **kwargs).copy()
            i_adj = np.sum(out * mean / std)
            out[intercept_idx] -= i_adj
            return out / std
        else:
            return algo(X, y, *args, **kwargs)
    return normalize_inputs


def sigmoid(x):
    """Sigmoid function of x."""
    return 1 / (1 + np.exp(-x))


@dispatch(object, object)
def dot(A, B):
    x = max([A, B], key=lambda x: getattr(x, '__array_priority__', 0))
    return np.dot(A, B)


@dispatch(da.Array, np.ndarray)
def dot(A, B):
    B = da.from_array(B, chunks=B.shape)
    return da.dot(A, B)


@dispatch(np.ndarray, da.Array)
def dot(A, B):
    A = da.from_array(A, chunks=A.shape)
    return da.dot(A, B)


@dispatch(np.ndarray, np.ndarray)
def dot(A, B):
    return np.dot(A, B)


@dispatch(da.Array, da.Array)
def dot(A, B):
    return da.dot(A, B)


@dispatch(object)
def sum(A):
    return A.sum()


@dispatch(object)
def add_intercept(X):
    return np.concatenate([X, np.ones_like(X, shape=(X.shape[0], 1))], axis=1)


@dispatch(da.Array)
def add_intercept(X):
    if np.isnan(np.sum(X.shape)):
        raise NotImplementedError("Can not add intercept to array with "
                                  "unknown chunk shape")
    j, k = X.chunks
    o = da.ones((X.shape[0], 1), chunks=(j, 1))
    # TODO: Needed this `.rechunk` for the solver to work
    # Is this OK / correct?
    X_i = da.concatenate([X, o], axis=1).rechunk((j, (k[0] + 1,)))
    return X_i


def make_y(X, beta=np.array([1.5, -3]), chunks=2):
    n, p = X.shape
    z0 = X.dot(beta)
    y = da.random.random(z0.shape, chunks=z0.chunks) < sigmoid(z0)
    return y

def mean_squared_error(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


def accuracy_score(y_true, y_pred):
    return (y_true == y_pred).mean()


def poisson_deviance(y_true, y_pred):
    return 2 * (y_true * np.log1p(y_true / y_pred) - (y_true - y_pred)).sum()


try:
    import sparse
except ImportError:
    pass
else:
    @dispatch(sparse.COO)
    def exp(x):
        return np.exp(x.todense())


def package_of(obj):
    """ Return package containing object's definition

    Or return None if not found
    """
    # http://stackoverflow.com/questions/43462701/get-package-of-python-object/43462865#43462865
    mod = inspect.getmodule(obj)
    if not mod:
        return
    base, _sep, _stem = mod.__name__.partition('.')
    return sys.modules[base]
