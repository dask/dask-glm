import pytest

import numpy as np

from dask_glm.logistic import proximal_logistic_loss, proximal_logistic_gradient, local_update

@pytest.mark.parametrize('N', [1000, 10000])
@pytest.mark.parametrize('beta',
        [np.array([-1.5, 3]),
         np.array([35, 2, 0, -3.2]),
         np.array([-1e-2, 1e-4, 1.0, 2e-3, -1.2])])
def test_local_update(N, beta):
    M = beta.shape[0]
    X = np.random.random((N, M))
    y = np.random.random(N)>0.4
    u = np.zeros(M)
    z = np.random.random(M)
    rho = 1e6

    result = local_update(X, y, beta, z, u, rho)

    assert np.allclose(result, z, atol=2e-3)
