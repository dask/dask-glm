import pytest
cupy = pytest.importorskip('cupy')

from dask import persist
import numpy as np

from dask_glm.algorithms import (newton, lbfgs, proximal_grad,
                                 gradient_descent, admm)
from dask_glm.families import Logistic, Normal, Poisson
from dask_glm.regularizers import Regularizer
from dask_glm.utils import sigmoid
from dask_glm.tests.cupy.utils import cupy_make_y


def add_l1(f, lam):
    def wrapped(beta, X, y):
        return f(beta, X, y) + lam * (np.abs(beta)).sum()
    return wrapped


def make_intercept_data(N, p, seed=20009):
    '''Given the desired number of observations (N) and
    the desired number of variables (p), creates
    random logistic data to test on.'''

    # set the seeds
    cupy.random.seed(seed)

    X = cupy.random.random((N, p + 1))
    col_sums = X.sum(axis=0)
    X = X / col_sums[None, :]
    X[:, p] = 1
    y = cupy_make_y(X, beta=cupy.random.random(p + 1))

    return X, y


@pytest.mark.parametrize('opt',
                         [lbfgs,
                          newton,
                          gradient_descent])
@pytest.mark.parametrize('N, p, seed',
                         [(100, 2, 20009),
                          (250, 12, 90210),
                          (95, 6, 70605)])
def test_methods(N, p, seed, opt):
    X, y = make_intercept_data(N, p, seed=seed)
    coefs = opt(X, y)
    p = sigmoid(X.dot(coefs))

    y_sum = y.sum()
    p_sum = p.sum()
    assert np.isclose(y_sum, p_sum, atol=1e-1)


@pytest.mark.parametrize('func,kwargs', [
    (newton, {'tol': 1e-5}),
    (lbfgs, {'tol': 1e-8}),
    (gradient_descent, {'tol': 1e-7}),
])
@pytest.mark.parametrize('N', [1000])
@pytest.mark.parametrize('nchunks', [1, 10])
@pytest.mark.parametrize('family', [Logistic, Normal, Poisson])
def test_basic_unreg_descent(func, kwargs, N, nchunks, family):
    beta = cupy.random.normal(size=2)
    M = len(beta)
    X = cupy.random.random((N, M))
    y = cupy_make_y(X, beta=cupy.array(beta))

    X, y = persist(X, y)

    result = func(X, y, family=family, **kwargs)
    test_vec = cupy.random.normal(size=2)

    opt = family.pointwise_loss(result, X, y)
    test_val = family.pointwise_loss(test_vec, X, y)

    assert opt < test_val


@pytest.mark.parametrize('func,kwargs', [
    (admm, {'abstol': 1e-4}),
    (proximal_grad, {'tol': 1e-7}),
])
@pytest.mark.parametrize('N', [1000])
@pytest.mark.parametrize('nchunks', [1, 10])
@pytest.mark.parametrize('family', [Logistic, Normal, Poisson])
@pytest.mark.parametrize('lam', [0.01, 1.2, 4.05])
@pytest.mark.parametrize('reg', [r() for r in Regularizer.__subclasses__()])
def test_basic_reg_descent(func, kwargs, N, nchunks, family, lam, reg):
    beta = cupy.random.normal(size=2)
    M = len(beta)
    X = cupy.random.random((N, M))
    y = cupy_make_y(X, beta=cupy.array(beta))

    X, y = persist(X, y)

    result = func(X, y, family=family, lamduh=lam, regularizer=reg, **kwargs)
    test_vec = cupy.random.normal(size=2)

    f = reg.add_reg_f(family.pointwise_loss, lam)

    opt = f(result, X, y)
    test_val = f(test_vec, X, y)

    assert opt < test_val


@pytest.mark.parametrize('func,kwargs', [
    (admm, {'max_iter': 2}),
    (proximal_grad, {'max_iter': 2}),
    (newton, {'max_iter': 2}),
    (gradient_descent, {'max_iter': 2}),
])
def test_determinism(func, kwargs):
    X, y = make_intercept_data(1000, 10)

    a = func(X, y, **kwargs)
    b = func(X, y, **kwargs)

    assert (a == b).all()


try:
    from distributed import Client
    from distributed.utils_test import cluster, loop  # flake8: noqa
except ImportError:
    pass
else:
    @pytest.mark.parametrize('func,kwargs', [
        (admm, {'max_iter': 2}),
        (proximal_grad, {'max_iter': 2}),
        (newton, {'max_iter': 2}),
        (gradient_descent, {'max_iter': 2}),
    ])
    def test_determinism_distributed(func, kwargs, loop):
        with cluster() as (s, [a, b]):
            with Client(s['address'], loop=loop) as c:
                X, y = make_intercept_data(1000, 10)

                a = func(X, y, **kwargs)
                b = func(X, y, **kwargs)

                assert (a == b).all()
