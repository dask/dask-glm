import pytest
cupy = pytest.importorskip('cupy')

import numpy as np
import cupy.testing as cpt
from dask_glm import regularizers as regs


@pytest.mark.parametrize('beta,expected', [
    (cupy.array([0, 0, 0]), 0),
    (cupy.array([1, 2, 3]), 7)
])
def test_l2_function(beta, expected):
    assert regs.L2().f(beta) == expected


@pytest.mark.parametrize('beta', [
    cupy.array([0, 0, 0]),
    cupy.array([1, 2, 3])
])
def test_l2_gradient(beta):
    cpt.assert_array_equal(regs.L2().gradient(beta), beta)


@pytest.mark.parametrize('beta', [
    cupy.array([0, 0, 0]),
    cupy.array([1, 2, 3])
])
def test_l2_hessian(beta):
    cpt.assert_array_equal(regs.L2().hessian(beta), np.eye(len(beta)))


@pytest.mark.parametrize('beta,expected', [
    (cupy.array([0, 0, 0]), cupy.array([0, 0, 0])),
    (cupy.array([1, 2, 3]), cupy.array([0.5, 1, 1.5]))
])
def test_l2_proximal_operator(beta, expected):
    cpt.assert_array_equal(regs.L2().proximal_operator(beta, 1), expected)


@pytest.mark.parametrize('beta,expected', [
    (cupy.array([0, 0, 0]), 0),
    (cupy.array([-1, 2, 3]), 6)
])
def test_l1_function(beta, expected):
    assert regs.L1().f(beta) == expected


@pytest.mark.parametrize('beta,expected', [
    (cupy.array([1, 2, 3]), cupy.array([1, 1, 1])),
    (cupy.array([-1, 2, 3]), cupy.array([-1, 1, 1]))
])
def test_l1_gradient(beta, expected):
    cpt.assert_array_equal(regs.L1().gradient(beta), expected)


@pytest.mark.parametrize('beta', [
    cupy.array([0.00000001, 1, 2]),
    cupy.array([-0.00000001, 1, 2]),
    cupy.array([0, 0, 0])
])
def test_l1_gradient_raises_near_zero(beta):
    with pytest.raises(ValueError):
        regs.L1().gradient(beta)


def test_l1_hessian():
    cpt.assert_array_equal(regs.L1().hessian(cupy.array([1, 2])),
                           cupy.array([[0, 0], [0, 0]]))


def test_l1_hessian_raises():
    with pytest.raises(ValueError):
        regs.L1().hessian(cupy.array([0, 0, 0]))


@pytest.mark.parametrize('beta,expected', [
    (cupy.array([0, 0, 0]), cupy.array([0, 0, 0])),
    (cupy.array([1, 2, 3]), cupy.array([0, 1, 2]))
])
def test_l1_proximal_operator(beta, expected):
    cpt.assert_array_equal(regs.L1().proximal_operator(beta, 1), expected)


@pytest.mark.parametrize('beta,expected', [
    (cupy.array([0, 0, 0]), 0),
    (cupy.array([1, 2, 3]), 6.5)
])
def test_elastic_net_function(beta, expected):
    assert regs.ElasticNet().f(beta) == expected


def test_elastic_net_function_zero_weight_is_l2():
    beta = cupy.array([1, 2, 3])
    assert regs.ElasticNet(weight=0).f(beta) == regs.L2().f(beta)


def test_elastic_net_function_zero_weight_is_l1():
    beta = cupy.array([1, 2, 3])
    assert regs.ElasticNet(weight=1).f(beta) == regs.L1().f(beta)


def test_elastic_net_gradient():
    beta = cupy.array([1, 2, 3])
    cpt.assert_array_equal(regs.ElasticNet(weight=0.5).gradient(beta), cupy.array([1, 1.5, 2]))


def test_elastic_net_gradient_zero_weight_is_l2():
    beta = cupy.array([1, 2, 3])
    cpt.assert_array_equal(regs.ElasticNet(weight=0).gradient(beta), regs.L2().gradient(beta))


def test_elastic_net_gradient_zero_weight_is_l1():
    beta = cupy.array([1, 2, 3])
    cpt.assert_array_equal(regs.ElasticNet(weight=1).gradient(beta), regs.L1().gradient(beta))


def test_elastic_net_hessian():
    beta = cupy.array([1, 2, 3])
    cpt.assert_array_equal(regs.ElasticNet(weight=0.5).hessian(beta),
                           np.eye(len(beta)) * regs.ElasticNet().weight)


def test_elastic_net_hessian_raises():
    with pytest.raises(ValueError):
        regs.ElasticNet(weight=0.5).hessian(cupy.array([0, 1, 2]))


def test_elastic_net_proximal_operator():
    beta = cupy.array([1, 2, 3])
    cpt.assert_array_equal(regs.ElasticNet(weight=0.5).proximal_operator(beta, 1), beta)
