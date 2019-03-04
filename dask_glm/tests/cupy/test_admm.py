import pytest

from dask import persist
import numpy as np
import cupy

from dask.array.utils import normalize_to_array
from dask_glm.algorithms import admm, local_update
from dask_glm.families import Logistic, Normal
from dask_glm.regularizers import L1
from dask_glm.utils import cupy_make_y


@pytest.mark.parametrize('N', [1000, 10000])
@pytest.mark.parametrize('beta',
                         [cupy.array([-1.5, 3]),
                          cupy.array([35, 2, 0, -3.2]),
                          cupy.array([-1e-2, 1e-4, 1.0, 2e-3, -1.2])])
@pytest.mark.parametrize('family', [Logistic, Normal])
def test_local_update(N, beta, family):
    M = beta.shape[0]
    X = cupy.random.random((N, M))
    y = cupy.random.random(N) > 0.4
    u = cupy.zeros(M)
    z = cupy.random.random(M)
    rho = 1e7

    def create_local_gradient(func):
        def wrapped(beta, X, y, z, u, rho):
            beta_like = np.empty_like(X, shape=beta.shape)
            beta_like[:] = beta
            return normalize_to_array(func(beta_like, X, y) + rho *
                                      (beta_like - z + u))
        return wrapped

    def create_local_f(func):
        def wrapped(beta, X, y, z, u, rho):
            beta_like = np.empty_like(X, shape=beta.shape)
            beta_like[:] = beta
            return normalize_to_array(func(beta_like, X, y) + (rho / 2) *
                                      np.dot(beta_like - z + u, beta_like - z + u))
        return wrapped

    f = create_local_f(family.pointwise_loss)
    fprime = create_local_gradient(family.pointwise_gradient)

    result = local_update(X, y, beta, z, u, rho, f=f, fprime=fprime)

    assert np.allclose(result, z, atol=2e-3)


@pytest.mark.parametrize('N', [1000, 10000])
@pytest.mark.parametrize('p', [1, 5, 10])
def test_admm_with_large_lamduh(N, p):
    X = cupy.random.random((N, p))
    beta = cupy.random.random(p)
    y = cupy_make_y(X, beta=cupy.array(beta))

    X, y = persist(X, y)
    z = admm(X, y, regularizer=L1(), lamduh=1e5, rho=20, max_iter=500)

    assert np.allclose(z, np.zeros(p), atol=1e-4)
