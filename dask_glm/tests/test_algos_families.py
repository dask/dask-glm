import dask
import dask.array as da
import dask.multiprocessing
import numpy as np
import pytest
from dask import persist

from dask_glm.algorithms import admm, gradient_descent, lbfgs, newton, proximal_grad
from dask_glm.families import Logistic, Normal, Poisson
from dask_glm.regularizers import Regularizer
from dask_glm.utils import make_y, maybe_to_cupy, sigmoid, to_dask_cupy_array_xy


def add_l1(f, lam):
    def wrapped(beta, X, y):
        return f(beta, X, y) + lam * (np.abs(beta)).sum()

    return wrapped


def make_intercept_data(N, p, seed=20009):
    """Given the desired number of observations (N) and
    the desired number of variables (p), creates
    random logistic data to test on."""

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


def make_array_type(type, *arrays):
    if type == "cupy":
        cupy = pytest.importorskip("cupy")
        return to_dask_cupy_array_xy(*arrays, cupy)
    elif type == "numpy":
        return tuple(a.compute() if hasattr(a, "compute") else a for a in arrays)
    else:
        return arrays


def maybe_compute(expr):
    return expr.compute() if hasattr(expr, "compute") else expr


@pytest.mark.parametrize("opt", [lbfgs, newton, gradient_descent])
@pytest.mark.parametrize(
    "N, p, seed,", [(100, 2, 20009), (250, 12, 90210), (95, 6, 70605)]
)
@pytest.mark.parametrize("array_type", ["dask", "numpy", "cupy"])
def test_methods(N, p, seed, opt, array_type):
    X, y = make_intercept_data(N, p, seed=seed)
    X, y = make_array_type(array_type, X, y)

    coefs = opt(X, y)
    dot = maybe_compute(X.dot(coefs))
    p = sigmoid(dot)

    y_sum = maybe_compute(y).sum()
    p_sum = p.sum()
    assert np.isclose(y_sum, p_sum, atol=1e-1)


@pytest.mark.parametrize(
    "func,kwargs",
    [
        (newton, {"tol": 1e-5}),
        (lbfgs, {"tol": 1e-8}),
        (gradient_descent, {"tol": 1e-7}),
    ],
)
@pytest.mark.parametrize("N", [1000])
@pytest.mark.parametrize("nchunks", [1, 10])
@pytest.mark.parametrize("family", [Logistic, Normal, Poisson])
@pytest.mark.parametrize("array_type", ["dask", "numpy", "cupy"])
def test_basic_unreg_descent(func, kwargs, N, nchunks, family, array_type):
    beta = np.random.normal(size=2)
    M = len(beta)
    X = da.random.random((N, M), chunks=(N // nchunks, M))
    y = make_y(X, beta=np.array(beta), chunks=(N // nchunks,))

    X, y = make_array_type(array_type, X, y)

    if array_type != "numpy":
        X, y = persist(X, y)

    result = func(X, y, family=family, **kwargs)
    test_vec = np.random.normal(size=2)
    test_vec = maybe_to_cupy(test_vec, X)

    opt = maybe_compute(family.pointwise_loss(result, X, y))
    test_val = maybe_compute(family.pointwise_loss(test_vec, X, y))

    assert opt < test_val


@pytest.mark.parametrize(
    "func,kwargs",
    [
        (admm, {"abstol": 1e-4}),
        (proximal_grad, {"tol": 1e-7}),
    ],
)
@pytest.mark.parametrize("N", [1000])
@pytest.mark.parametrize("nchunks", [1, 10])
@pytest.mark.parametrize("family", [Logistic, Normal, Poisson])
@pytest.mark.parametrize("lam", [0.01, 1.2, 4.05])
@pytest.mark.parametrize("reg", [r() for r in Regularizer.__subclasses__()])
@pytest.mark.parametrize("array_type", ["dask", "numpy", "cupy"])
def test_basic_reg_descent(func, kwargs, N, nchunks, family, lam, reg, array_type):
    beta = np.random.normal(size=2)
    M = len(beta)
    X = da.random.random((N, M), chunks=(N // nchunks, M))
    y = make_y(X, beta=np.array(beta), chunks=(N // nchunks,))

    X, y = make_array_type(array_type, X, y)

    if array_type != "numpy":
        X, y = persist(X, y)

    result = func(X, y, family=family, lamduh=lam, regularizer=reg, **kwargs)
    test_vec = np.random.normal(size=2)
    test_vec = maybe_to_cupy(test_vec, X)

    f = reg.add_reg_f(family.pointwise_loss, lam)

    opt = maybe_compute(f(result, X, y))
    test_val = maybe_compute(f(test_vec, X, y))

    assert opt < test_val


@pytest.mark.parametrize(
    "func,kwargs",
    [
        (admm, {"max_iter": 2}),
        (proximal_grad, {"max_iter": 2}),
        (newton, {"max_iter": 2}),
        (gradient_descent, {"max_iter": 2}),
    ],
)
@pytest.mark.parametrize("scheduler", ["synchronous", "threading", "multiprocessing"])
@pytest.mark.parametrize("array_type", ["dask", "numpy", "cupy"])
def test_determinism(func, kwargs, scheduler, array_type):
    X, y = make_intercept_data(1000, 10)
    X, y = make_array_type(array_type, X, y)

    with dask.config.set(scheduler=scheduler):
        a = func(X, y, **kwargs)
        b = func(X, y, **kwargs)

    assert (a == b).all()


try:
    from distributed import Client
    from distributed.utils_test import cleanup, cluster, loop_in_thread  # noqa: F401
except ImportError:
    pass
else:

    @pytest.mark.parametrize(
        "func,kwargs",
        [
            (admm, {"max_iter": 2}),
            (proximal_grad, {"max_iter": 2}),
            (newton, {"max_iter": 2}),
            (gradient_descent, {"max_iter": 2}),
        ],
    )
    def test_determinism_distributed(func, kwargs, loop_in_thread):  # noqa: F811
        with cluster() as (s, [a, b]):
            with Client(s["address"], loop=loop_in_thread) as _:
                X, y = make_intercept_data(1000, 10)

                a = func(X, y, **kwargs)
                b = func(X, y, **kwargs)

                assert (a == b).all()

    def test_broadcast_lbfgs_weight(loop_in_thread):  # noqa: F811
        with cluster() as (s, [a, b]):
            with Client(s["address"], loop=loop_in_thread) as c:
                X, y = make_intercept_data(1000, 10)
                coefs = lbfgs(X, y, dask_distributed_client=c)
                p = sigmoid(X.dot(coefs).compute())

                y_sum = y.compute().sum()
                p_sum = p.sum()
                assert np.isclose(y_sum, p_sum, atol=1e-1)
