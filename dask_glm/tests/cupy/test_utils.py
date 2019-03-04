import pytest
import numpy as np
import cupy
import cupy.testing as cpt

from dask_glm import utils
from dask.array.utils import assert_eq


def test_normalize_normalizes():
    @utils.normalize
    def do_nothing(X, y):
        return cupy.array([0.0, 1.0, 2.0])
    X = cupy.array([[1, 0, 0], [1, 2, 2]])
    y = cupy.array([0, 1, 0])
    res = do_nothing(X, y)
    cpt.assert_array_equal(res, cupy.array([-3.0, 1.0, 2.0]))


def test_normalize_doesnt_normalize():
    @utils.normalize
    def do_nothing(X, y):
        return cupy.array([0.0, 1.0, 2.0])
    X = cupy.array([[1, 0, 0], [1, 2, 2]])
    y = cupy.array([0, 1, 0])
    res = do_nothing(X, y, normalize=False)
    cpt.assert_array_equal(res, cupy.array([0, 1, 2]))


def test_normalize_normalizes_if_intercept_not_present():
    @utils.normalize
    def do_nothing(X, y):
        return cupy.array([0.0, 1.0, 2.0])
    X = cupy.array([[1, 0, 0], [3, 9.0, 2]])
    y = cupy.array([0, 1, 0])
    res = do_nothing(X, y)
    cpt.assert_array_equal(res, cupy.array([0, 1 / 4.5, 2]))


def test_normalize_raises_if_multiple_constants():
    @utils.normalize
    def do_nothing(X, y):
        return cupy.array([0.0, 1.0, 2.0])
    X = cupy.array([[1, 2, 3], [1, 2, 3]])
    y = cupy.array([0, 1, 0])
    with pytest.raises(ValueError):
        res = do_nothing(X, y)


def test_add_intercept():
    X = cupy.zeros((4, 4))
    result = utils.add_intercept(X)
    expected = cupy.array([
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
    ], dtype=X.dtype)
    cpt.assert_array_equal(result, expected)
