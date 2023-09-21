import dask.array as da
import numpy as np
import pytest
import sparse
from dask.array.utils import assert_eq

from dask_glm import utils


def test_normalize_normalizes():
    @utils.normalize
    def do_nothing(X, y):
        return np.array([0.0, 1.0, 2.0])

    X = da.from_array(np.array([[1, 0, 0], [1, 2, 2]]), chunks=(2, 3))
    y = da.from_array(np.array([0, 1, 0]), chunks=(3,))
    res = do_nothing(X, y)
    np.testing.assert_equal(res, np.array([-3.0, 1.0, 2.0]))


def test_normalize_doesnt_normalize():
    @utils.normalize
    def do_nothing(X, y):
        return np.array([0.0, 1.0, 2.0])

    X = da.from_array(np.array([[1, 0, 0], [1, 2, 2]]), chunks=(2, 3))
    y = da.from_array(np.array([0, 1, 0]), chunks=(3,))
    res = do_nothing(X, y, normalize=False)
    np.testing.assert_equal(res, np.array([0, 1, 2]))


def test_normalize_normalizes_if_intercept_not_present():
    @utils.normalize
    def do_nothing(X, y):
        return np.array([0.0, 1.0, 2.0])

    X = da.from_array(np.array([[1, 0, 0], [3, 9.0, 2]]), chunks=(2, 3))
    y = da.from_array(np.array([0, 1, 0]), chunks=(3,))
    res = do_nothing(X, y)
    np.testing.assert_equal(res, np.array([0, 1 / 4.5, 2]))


def test_normalize_raises_if_multiple_constants():
    @utils.normalize
    def do_nothing(X, y):
        return np.array([0.0, 1.0, 2.0])

    X = da.from_array(np.array([[1, 2, 3], [1, 2, 3]]), chunks=(2, 3))
    y = da.from_array(np.array([0, 1, 0]), chunks=(3,))
    with pytest.raises(ValueError):
        do_nothing(X, y)


def test_add_intercept():
    X = np.zeros((4, 4))
    result = utils.add_intercept(X)
    expected = np.array(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
        ],
        dtype=X.dtype,
    )
    assert_eq(result, expected)


def test_add_intercept_dask():
    X = da.from_array(np.zeros((4, 4)), chunks=(2, 4))
    result = utils.add_intercept(X)
    expected = da.from_array(
        np.array(
            [
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
            ],
            dtype=X.dtype,
        ),
        chunks=2,
    )
    assert_eq(result, expected)


def test_add_intercept_sparse():
    X = sparse.COO(np.zeros((4, 4)))
    result = utils.add_intercept(X)
    expected = sparse.COO(
        np.array(
            [
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
            ],
            dtype=X.dtype,
        )
    )
    assert (result == expected).all()


@pytest.mark.xfail(
    reason=(
        "TODO: ValueError: This operation requires consistent fill-values, "
        "but argument 1 had a fill value of 1.0, which is different from a "
        "fill_value of 0.0 in the first argument."
    )
)
def test_add_intercept_sparse_dask():
    X = da.from_array(sparse.COO(np.zeros((4, 4))), chunks=(2, 4))
    result = utils.add_intercept(X)
    expected = da.from_array(
        sparse.COO(
            np.array(
                [
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1],
                ],
                dtype=X.dtype,
            )
        ),
        chunks=2,
    )
    assert_eq(result, expected)


def test_sparse():
    x = sparse.COO({(0, 0): 1, (1, 2): 2, (2, 1): 3})
    y = x.todense()
    assert utils.sum(x) == utils.sum(x.todense())
    for func in [utils.sigmoid, utils.sum, utils.exp]:
        assert (func(x) == func(y)).all()


def test_dask_array_is_sparse():
    assert utils.is_dask_array_sparse(da.from_array(sparse.COO([], [], shape=(10, 10))))
    assert utils.is_dask_array_sparse(da.from_array(sparse.eye(10)))
    assert not utils.is_dask_array_sparse(da.from_array(np.eye(10)))


@pytest.mark.xfail(
    reason="dask does not forward DOK in _meta "
    "(https://github.com/pydata/sparse/issues/292)"
)
def test_dok_dask_array_is_sparse():
    assert utils.is_dask_array_sparse(da.from_array(sparse.DOK((10, 10))))


def test_dot_with_cupy():
    cupy = pytest.importorskip("cupy")

    # dot(cupy.array, cupy.array)
    A = cupy.random.rand(100, 100)
    B = cupy.random.rand(100)
    ans = cupy.dot(A, B)
    res = utils.dot(A, B)
    assert_eq(ans, res)

    # dot(dask.array, cupy.array)
    dA = da.from_array(A, chunks=(10, 100))
    res = utils.dot(dA, B).compute()
    assert_eq(ans, res)

    # dot(cupy.array, dask.array)
    dB = da.from_array(B, chunks=(10))
    res = utils.dot(A, dB).compute()
    assert_eq(ans, res)


def test_dot_with_sparse():
    A = sparse.random((1024, 64))
    B = sparse.random((64))
    ans = sparse.dot(A, B)

    # dot(sparse.array, sparse.array)
    res = utils.dot(A, B)
    assert_eq(ans, res)

    # dot(sparse.array, dask.array)
    res = utils.dot(A, da.from_array(B, chunks=B.shape))
    assert_eq(ans, res.compute())

    # dot(dask.array, sparse.array)
    res = utils.dot(da.from_array(A, chunks=A.shape), B)
    assert_eq(ans, res.compute())
