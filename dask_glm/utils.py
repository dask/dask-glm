from __future__ import absolute_import, division, print_function

import inspect
import sys
from functools import wraps

import dask.array as da
import numpy as np
import sparse
from dask.distributed import get_client
from multipledispatch import dispatch


def normalize(algo):
    @wraps(algo)
    def normalize_inputs(X, y, *args, **kwargs):
        normalize = kwargs.pop("normalize", True)
        if normalize:
            mean, std = da.compute(X.mean(axis=0), X.std(axis=0))
            mean, std = mean.copy(), std.copy()  # in case they are read-only
            intercept_idx = np.where(std == 0)
            if len(intercept_idx[0]) > 1:
                raise ValueError("Multiple constant columns detected!")
            mean[intercept_idx] = 0
            std[intercept_idx] = 1
            mean = (
                mean
                if len(intercept_idx[0])
                else zeros(mean.shape, getattr(X, "_meta", None))
            )
            Xn = (X - mean) / std
            out = algo(Xn, y, *args, **kwargs).copy()
            i_adj = np.sum(out * mean / std)
            out[intercept_idx] -= i_adj
            return out / std
        else:
            return algo(X, y, *args, **kwargs)

    return normalize_inputs


def zeros(shape, arr=None, **kwargs):
    if arr:
        return np.zeros_like(arr, shape=shape, **kwargs)
    else:
        return np.zeros(shape=shape, **kwargs)


def sigmoid(x):
    """Sigmoid function of x."""
    return 1 / (1 + exp(-x))


@dispatch(object)
def exp(A):
    return np.exp(A)


@dispatch(float)
def exp(A):
    return np.exp(A)


@dispatch(np.ndarray)
def exp(A):
    return np.exp(A)


@dispatch(da.Array)
def exp(A):
    return da.exp(A)


@dispatch(object)
def absolute(A):
    return abs(A)


@dispatch(np.ndarray)
def absolute(A):
    return np.absolute(A)


@dispatch(da.Array)
def absolute(A):
    return da.absolute(A)


@dispatch(object)
def sign(A):
    return A.sign()


@dispatch(np.ndarray)
def sign(A):
    return np.sign(A)


@dispatch(da.Array)
def sign(A):
    return da.sign(A)


@dispatch(object)
def log1p(A):
    return np.log1p(A)


@dispatch(np.ndarray)
def log1p(A):
    return np.log1p(A)


@dispatch(da.Array)
def log1p(A):
    return da.log1p(A)


@dispatch(object, object)
def dot(A, B):
    if isinstance(A, da.Array) or isinstance(B, da.Array):
        A = da.asarray(A, chunks=A.shape)
        B = da.asarray(B, chunks=B.shape)
    return np.dot(A, B)


@dispatch(object)
def sum(A):
    return A.sum()


def is_dask_array_sparse(X):
    """
    Check using _meta if a dask array contains sparse arrays
    """
    return isinstance(X._meta, sparse.SparseArray)


@dispatch(np.ndarray)
def add_intercept(X):
    return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)


@dispatch(sparse.SparseArray)
def add_intercept(X):
    return sparse.concatenate([X, sparse.COO(np.ones((X.shape[0], 1)))], axis=1)


@dispatch(da.Array)
def add_intercept(X):
    if np.isnan(np.sum(X.shape)):
        raise NotImplementedError(
            "Can not add intercept to array with " "unknown chunk shape"
        )
    j, k = X.chunks
    o = da.ones_like(X, shape=(X.shape[0], 1), chunks=(j, 1))
    if is_dask_array_sparse(X):
        o = o.map_blocks(sparse.COO)
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
    return 2 * (y_true * log1p(y_true / y_pred) - (y_true - y_pred)).sum()


try:
    import sparse
except ImportError:
    pass
else:

    @dispatch(sparse.COO)
    def exp(x):
        return np.exp(x.todense())


def package_of(obj):
    """Return package containing object's definition

    Or return None if not found
    """
    # http://stackoverflow.com/questions/43462701/get-package-of-python-object/43462865#43462865
    mod = inspect.getmodule(obj)
    if not mod:
        return
    base, _sep, _stem = mod.__name__.partition(".")
    return sys.modules[base]


def scatter_array(arr, dask_client):
    """Scatter a large numpy array into workers
    Return the equivalent dask array
    """
    future_arr = dask_client.scatter(arr)
    return da.from_delayed(
        future_arr, shape=arr.shape, dtype=arr.dtype, meta=np.zeros_like(arr, shape=())
    )


def get_distributed_client():
    try:
        return get_client()
    except ValueError:
        return None


def maybe_to_cupy(beta, X):
    """convert beta, a numpy array, to a cupy array
    if X is a cupy array or dask cupy array
    """
    if "cupy" in str(type(X)) or hasattr(X, "_meta") and "cupy" in str(type(X._meta)):
        import cupy

        return cupy.asarray(beta)
    return beta


def to_dask_cupy_array(X, cupy):
    """convert a dask numpy array to a dask cupy array"""
    return X.map_blocks(
        lambda x: cupy.asarray(x), dtype=X.dtype, meta=cupy.asarray(X._meta)
    )


def to_dask_cupy_array_xy(X, y, cupy):
    return to_dask_cupy_array(X, cupy), to_dask_cupy_array(y, cupy)
