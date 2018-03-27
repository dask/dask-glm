"""Optimization algorithms for solving minimizaiton problems.
"""

from __future__ import absolute_import, division, print_function
import time
from warnings import warn

from dask import delayed, persist, compute, set_options
import functools
import numpy as np
import numpy.linalg as LA
import dask.array as da
import dask.dataframe as dd
from scipy.optimize import fmin_l_bfgs_b


from dask_glm.utils import dot, normalize
from dask_glm.families import Logistic
from dask_glm.regularizers import Regularizer


def compute_stepsize_dask(beta, step, Xbeta, Xstep, y, curr_val,
                          family=Logistic, stepSize=1.0,
                          armijoMult=0.1, backtrackMult=0.1):
    """Compute the optimal stepsize

    Parameters
    ----------
    beta : array-like
    step : float
    XBeta : array-lie
    Xstep :
    y : array-like
    curr_val : float
    famlily : Family, optional
    stepSize : float, optional
    armijoMult : float, optional
    backtrackMult : float, optional

    Returns
    -------
    stepSize : flaot
    beta : array-like
    xBeta : array-like
    func : callable
    """

    loglike = family.loglike
    beta, step, Xbeta, Xstep, y, curr_val = persist(beta, step, Xbeta, Xstep, y, curr_val)
    obeta, oXbeta = beta, Xbeta
    (step,) = compute(step)
    steplen = (step ** 2).sum()
    lf = curr_val
    func = 0
    for ii in range(100):
        beta = obeta - stepSize * step
        if ii and (beta == obeta).all():
            stepSize = 0
            break

        Xbeta = oXbeta - stepSize * Xstep
        func = loglike(Xbeta, y)
        Xbeta, func = persist(Xbeta, func)

        df = lf - compute(func)[0]
        if df >= armijoMult * stepSize * steplen:
            break
        stepSize *= backtrackMult

    return stepSize, beta, Xbeta, func


@normalize
def gradient_descent(X, y, max_iter=100, tol=1e-14, family=Logistic, **kwargs):
    """
    Michael Grant's implementation of Gradient Descent.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    max_iter : int
        maximum number of iterations to attempt before declaring
        failure to converge
    tol : float
        Maximum allowed change from prior iteration required to
        declare convergence
    family : Family

    Returns
    -------
    beta : array-like, shape (n_features,)
    """
    loglike, gradient = family.loglike, family.gradient
    n, p = X.shape
    firstBacktrackMult = 0.1
    nextBacktrackMult = 0.5
    armijoMult = 0.1
    stepGrowth = 1.25
    stepSize = 1.0
    recalcRate = 10
    backtrackMult = firstBacktrackMult
    beta = np.zeros(p)

    for k in range(max_iter):
        # how necessary is this recalculation?
        if k % recalcRate == 0:
            Xbeta = X.dot(beta)
            func = loglike(Xbeta, y)

        grad = gradient(Xbeta, X, y)
        Xgradient = X.dot(grad)

        # backtracking line search
        lf = func
        stepSize, _, _, func = compute_stepsize_dask(beta, grad,
                                                     Xbeta, Xgradient,
                                                     y, func, family=family,
                                                     backtrackMult=backtrackMult,
                                                     armijoMult=armijoMult,
                                                     stepSize=stepSize)

        beta, stepSize, Xbeta, lf, func, grad, Xgradient = persist(
            beta, stepSize, Xbeta, lf, func, grad, Xgradient)

        stepSize, lf, func, grad = compute(stepSize, lf, func, grad)

        beta = beta - stepSize * grad  # tiny bit of repeat work here to avoid communication
        Xbeta = Xbeta - stepSize * Xgradient

        if stepSize == 0:
            break

        df = lf - func
        df /= max(func, lf)

        if df < tol:
            break
        stepSize *= stepGrowth
        backtrackMult = nextBacktrackMult

    return beta


def _get_n(n, kwargs):
    if np.isnan(n):
        if 'n' in kwargs:
            return kwargs.get('n')
        raise ValueError('Could not get the number of examples `n`. Pass the '
                         'number of examples in as a keyword argument: '
                         '`sgd(..., n=n)`. If using a distributed dataframe, '
                         '`sgd(..., n=len(ddf))` works')
    return n


def _shuffle_blocks(x, seed=42):
    rng = np.random.RandomState(seed)
    i = rng.permutation(x.shape[0]).astype(int)
    y = x[i]
    return y


@normalize
def sgd(X, y, epochs=100, tol=1e-3, family=Logistic, batch_size=64,
        initial_step=1e-4, callback=None, average=True, maxiter=np.inf, **kwargs):
    r"""Stochastic Gradient Descent.

    Parameters
    ----------
    X: array - like, shape(n_samples, n_features)
    y: array - like, shape(n_samples,)
    epochs: int, float
        maximum number of passes through the dataset
    tol: float
        Maximum allowed change from prior iteration required to
        declare convergence
    batch_size: int
        The batch size used to approximate the gradient. Larger batch sizes
        will approximate the gradient better.
    initial_step: float
        The initial step size. The step size is decays like 1 / k.
    callback: callable
        A callback to call every iteration that accepts keyword arguments
        `X`, `y`, `beta`, `grad`, `nit` (number of iterations) and `family`
    average: bool
        To average the parameters found or not. See[1]_.
    family: Family

    Returns
    -------
    beta : array-like, shape (n_features,)

    Notes
    -----

    The current implementation assumes that the dataset is "well shuffled", or
    each block is a representative example of the gradient. More formally, we
    assume the gradient approximation is an unbiased approximation regardless
    of which block is sampled.

    .. _1: https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Averaging
    """
    gradient = family.gradient
    n, p = X.shape
    n = _get_n(n, kwargs)

    beta = np.zeros(p)
    if average:
        beta_sum = np.zeros(p)

    nit = 0

    # step_size = O(1/sqrt(k)) from "Non-asymptotic analysis of
    # stochastic approximation algorithms for machine learning" by
    # Moulines, Eric and Bach, Francis Rsgd
    # but, this may require many iterations. Using
    # step_size = lambda init, nit, decay: init * decay**(nit//n)
    # is used in practice but not testing now
    step_size = lambda init, nit: init / np.sqrt(nit + 1)
    while True:
        seed = int(time.time() * 1000) % 2**32  # millisecond timings
        X = X.map_blocks(_shuffle_blocks, seed=seed, dtype=X.dtype)
        y = y.map_blocks(_shuffle_blocks, seed=seed, dtype=y.dtype)
        for k in range(n // batch_size):
            beta_old = beta.copy()
            nit += 1

            start = np.random.choice(n - batch_size)
            i = slice(start, start + batch_size)
            Xbeta = dot(X[i], beta)
            grad = gradient(Xbeta, X[i], y[i]).compute()

            beta -= step_size(initial_step, nit) * (n / batch_size) * grad
            if average:
                beta_sum += beta
            if callback:
                callback(X=X[i], y=y[i], grad=grad, nit=nit, family=family,
                         beta=beta if not average else beta_sum / nit)

            rel_error = LA.norm(beta_old - beta) / LA.norm(beta)
            converged = (rel_error < tol) or (nit / n > epochs) or (nit > maxiter)
            if converged:
                break
    if average:
        return beta_sum / nit
    return beta


@normalize
def newton(X, y, max_iter=50, tol=1e-8, family=Logistic, **kwargs):
    """Newtons Method for Logistic Regression.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    max_iter : int
        maximum number of iterations to attempt before declaring
        failure to converge
    tol : float
        Maximum allowed change from prior iteration required to
        declare convergence
    family : Family

    Returns
    -------
    beta : array-like, shape (n_features,)
    """
    gradient, hessian = family.gradient, family.hessian
    n, p = X.shape
    beta = np.zeros(p)  # always init to zeros?
    Xbeta = dot(X, beta)

    iter_count = 0
    converged = False

    while not converged:
        beta_old = beta

        # should this use map_blocks()?
        hess = hessian(Xbeta, X)
        grad = gradient(Xbeta, X, y)

        hess, grad = da.compute(hess, grad)

        # should this be dask or numpy?
        # currently uses Python 3 specific syntax
        step, _, _, _ = np.linalg.lstsq(hess, grad)
        beta = (beta_old - step)

        iter_count += 1

        # should change this criterion
        coef_change = np.absolute(beta_old - beta)
        converged = (
            (not np.any(coef_change > tol)) or (iter_count > max_iter))

        if not converged:
            Xbeta = dot(X, beta)  # numpy -> dask converstion of beta

    return beta


@normalize
def admm(X, y, regularizer='l1', lamduh=0.1, rho=1, over_relax=1,
         max_iter=250, abstol=1e-4, reltol=1e-2, family=Logistic, **kwargs):
    """
    Alternating Direction Method of Multipliers

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    regularizer : str or Regularizer
    lambuh : float
    rho : float
    over_relax : FLOAT
    max_iter : int
        maximum number of iterations to attempt before declaring
        failure to converge
    abstol, reltol : float
    family : Family

    Returns
    -------
    beta : array-like, shape (n_features,)
    """
    pointwise_loss = family.pointwise_loss
    pointwise_gradient = family.pointwise_gradient
    regularizer = Regularizer.get(regularizer)

    def create_local_gradient(func):
        @functools.wraps(func)
        def wrapped(beta, X, y, z, u, rho):
            return func(beta, X, y) + rho * (beta - z + u)
        return wrapped

    def create_local_f(func):
        @functools.wraps(func)
        def wrapped(beta, X, y, z, u, rho):
            return func(beta, X, y) + (rho / 2) * np.dot(beta - z + u,
                                                         beta - z + u)
        return wrapped

    f = create_local_f(pointwise_loss)
    fprime = create_local_gradient(pointwise_gradient)

    nchunks = getattr(X, 'npartitions', 1)
    # nchunks = X.npartitions
    (n, p) = X.shape
    # XD = X.to_delayed().flatten().tolist()
    # yD = y.to_delayed().flatten().tolist()
    if isinstance(X, da.Array):
        XD = X.rechunk((None, X.shape[-1])).to_delayed().flatten().tolist()
    else:
        XD = [X]
    if isinstance(y, da.Array):
        yD = y.rechunk((None, y.shape[-1])).to_delayed().flatten().tolist()
    else:
        yD = [y]

    z = np.zeros(p)
    u = np.array([np.zeros(p) for i in range(nchunks)])
    betas = np.array([np.ones(p) for i in range(nchunks)])

    for k in range(max_iter):

        # x-update step
        new_betas = [delayed(local_update)(xx, yy, bb, z, uu, rho, f=f,
                                           fprime=fprime) for
                     xx, yy, bb, uu in zip(XD, yD, betas, u)]
        new_betas = np.array(da.compute(*new_betas))

        beta_hat = over_relax * new_betas + (1 - over_relax) * z

        #  z-update step
        zold = z.copy()
        ztilde = np.mean(beta_hat + np.array(u), axis=0)
        z = regularizer.proximal_operator(ztilde, lamduh / (rho * nchunks))

        # u-update step
        u += beta_hat - z

        # check for convergence
        primal_res = np.linalg.norm(new_betas - z)
        dual_res = np.linalg.norm(rho * (z - zold))

        eps_pri = np.sqrt(p * nchunks) * abstol + reltol * np.maximum(
            np.linalg.norm(new_betas), np.sqrt(nchunks) * np.linalg.norm(z))
        eps_dual = np.sqrt(p * nchunks) * abstol + \
            reltol * np.linalg.norm(rho * u)

        if primal_res < eps_pri and dual_res < eps_dual:
            break

    return z


def local_update(X, y, beta, z, u, rho, f, fprime, solver=fmin_l_bfgs_b):

    beta = beta.ravel()
    u = u.ravel()
    z = z.ravel()
    solver_args = (X, y, z, u, rho)
    beta, f, d = solver(f, beta, fprime=fprime, args=solver_args,
                        maxiter=200,
                        maxfun=250)

    return beta


@normalize
def lbfgs(X, y, regularizer=None, lamduh=1.0, max_iter=100, tol=1e-4,
          family=Logistic, verbose=False, **kwargs):
    """L-BFGS solver using scipy.optimize implementation

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    max_iter : int
        maximum number of iterations to attempt before declaring
        failure to converge
    tol : float
        Maximum allowed change from prior iteration required to
        declare convergence
    family : Family

    Returns
    -------
    beta : array-like, shape (n_features,)
    """
    pointwise_loss = family.pointwise_loss
    pointwise_gradient = family.pointwise_gradient
    if regularizer is not None:
        regularizer = Regularizer.get(regularizer)
        pointwise_loss = regularizer.add_reg_f(pointwise_loss, lamduh)
        pointwise_gradient = regularizer.add_reg_grad(pointwise_gradient, lamduh)

    n, p = X.shape
    beta0 = np.zeros(p)

    def compute_loss_grad(beta, X, y):
        loss_fn = pointwise_loss(beta, X, y)
        gradient_fn = pointwise_gradient(beta, X, y)
        loss, gradient = compute(loss_fn, gradient_fn)
        return loss, gradient.copy()

    with set_options(fuse_ave_width=0):  # optimizations slows this down
        beta, loss, info = fmin_l_bfgs_b(
            compute_loss_grad, beta0, fprime=None,
            args=(X, y),
            iprint=(verbose > 0) - 1, pgtol=tol, maxiter=max_iter)

    return beta


@normalize
def proximal_grad(X, y, regularizer='l1', lamduh=0.1, family=Logistic,
                  max_iter=100, tol=1e-8, **kwargs):
    """

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    max_iter : int
        maximum number of iterations to attempt before declaring
        failure to converge
    tol : float
        Maximum allowed change from prior iteration required to
        declare convergence
    family : Family
    verbose : bool, default False
        whether to print diagnostic information during convergence

    Returns
    -------
    beta : array-like, shape (n_features,)
    """
    n, p = X.shape
    firstBacktrackMult = 0.1
    nextBacktrackMult = 0.5
    armijoMult = 0.1
    stepGrowth = 1.25
    stepSize = 1.0
    recalcRate = 10
    backtrackMult = firstBacktrackMult
    beta = np.zeros(p)
    regularizer = Regularizer.get(regularizer)

    for k in range(max_iter):
        # Compute the gradient
        if k % recalcRate == 0:
            Xbeta = X.dot(beta)
            func = family.loglike(Xbeta, y)

        gradient = family.gradient(Xbeta, X, y)

        Xbeta, func, gradient = persist(
            Xbeta, func, gradient)

        obeta = beta

        # Compute the step size
        lf = func
        for ii in range(100):
            beta = regularizer.proximal_operator(obeta - stepSize * gradient, stepSize * lamduh)
            step = obeta - beta
            Xbeta = X.dot(beta)

            Xbeta, beta = persist(Xbeta, beta)

            func = family.loglike(Xbeta, y)
            func = persist(func)[0]
            func = compute(func)[0]
            df = lf - func
            if df > 0:
                break
            stepSize *= backtrackMult
        if stepSize == 0:
            break
        df /= max(func, lf)
        if df < tol:
            break
        stepSize *= stepGrowth
        backtrackMult = nextBacktrackMult

    # L2-regularization returned a dask-array
    try:
        return beta.compute()
    except AttributeError:
        return beta


_solvers = {
    'admm': admm,
    'gradient_descent': gradient_descent,
    'newton': newton,
    'lbfgs': lbfgs,
    'proximal_grad': proximal_grad,
    'sgd': sgd
}
