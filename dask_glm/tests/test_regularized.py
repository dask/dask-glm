from __future__ import absolute_import, division, print_function

import dask.array as da
import numpy as np
import pytest

from dask_glm.logistic import proximal_grad
from dask_glm.utils import make_y


def make_data(N, p, seed=20009):
    '''Given the desired number of observations (N) and
    the desired number of variables (p), creates
    random logistic data to test on.'''

    # set the seeds
    da.random.seed(seed)
    np.random.seed(seed)

    X = np.random.random((N, p + 1))
    col_sums = X.sum(axis=0)
    X = X / col_sums[None, :]
    X[:, p] = 1
    X = da.from_array(X, chunks=(N / 5, p + 1))
    y = make_y(X, beta=np.random.random(p + 1))

    return X, y


@pytest.mark.parametrize('opt', [proximal_grad])
@pytest.mark.parametrize('N, p, seed',
                         [(100, 2, 20009),
                          (250, 12, 90210),
                          (95, 6, 70605)])
def test_l1(N, p, seed, opt):
    X, y = make_data(N, p, seed=seed)
    coefs = opt(X, y, reg='l1', lamduh=1000)
    assert np.all(np.isclose(coefs, 0))
