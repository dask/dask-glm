import numpy as np
import numpy.testing as npt
import pytest
from dask_glm import regularizers as regs


@pytest.mark.parametrize('func,args', [
    ('f', [0]),
    ('gradient', [0]),
    ('hessian', [0]),
    ('proximal_operator', [0, 1])
])
def test_base_class_raises_notimplementederror(func, args):
    with pytest.raises(NotImplementedError):
        getattr(regs.Regularizer(), func)(*args)


class FooRegularizer(regs.Regularizer):

    def f(self, beta):
        return beta + 1

    def gradient(self, beta):
        return beta + 1

    def hessian(self, beta):
        return beta + 1


@pytest.mark.parametrize('func', [
    'add_reg_f',
    'add_reg_grad',
    'add_reg_hessian'
])
def test_add_reg_funcs(func):
    def foo(x):
        return x**2
    new_func = getattr(FooRegularizer(), func)(foo, 1)
    assert callable(new_func)
    assert new_func(2) == 7


@pytest.mark.parametrize('beta,expected', [
    (np.array([0, 0, 0]), 0),
    (np.array([1, 2, 3]), 7)
])
def test_l2_function(beta, expected):
    assert regs.L2().f(beta) == expected


@pytest.mark.parametrize('beta', [
    np.array([0, 0, 0]),
    np.array([1, 2, 3])
])
def test_l2_gradient(beta):
    npt.assert_array_equal(regs.L2().gradient(beta), beta)


@pytest.mark.parametrize('beta', [
    np.array([0, 0, 0]),
    np.array([1, 2, 3])
])
def test_l2_hessian(beta):
    npt.assert_array_equal(regs.L2().hessian(beta), np.eye(len(beta)))


@pytest.mark.parametrize('beta,expected', [
    (np.array([0, 0, 0]), np.array([0, 0, 0])),
    (np.array([1, 2, 3]), np.array([0.5, 1, 1.5]))
])
def test_l2_proximal_operator(beta, expected):
    npt.assert_array_equal(regs.L2().proximal_operator(beta, 1), expected)


@pytest.mark.parametrize('beta,expected', [
    (np.array([0, 0, 0]), 0),
    (np.array([-1, 2, 3]), 6)
])
def test_l1_function(beta, expected):
    assert regs.L1().f(beta) == expected


@pytest.mark.parametrize('beta,expected', [
    (np.array([1, 2, 3]), np.array([1, 1, 1])),
    (np.array([-1, 2, 3]), np.array([-1, 1, 1]))
])
def test_l1_gradient(beta, expected):
    npt.assert_array_equal(regs.L1().gradient(beta), expected)


@pytest.mark.parametrize('beta', [
    np.array([0.00000001, 1, 2]),
    np.array([-0.00000001, 1, 2]),
    np.array([0, 0, 0])
])
def test_l1_gradient_raises_near_zero(beta):
    with pytest.raises(ValueError):
        regs.L1().gradient(beta)


@pytest.mark.parametrize('beta', [
    np.array([0, 0, 0]),
    np.array([1, 2, 3])
])
def test_l1_hessian_raises(beta):
    with pytest.raises(ValueError):
        regs.L1().hessian(beta)


@pytest.mark.parametrize('beta,expected', [
    (np.array([0, 0, 0]), np.array([0, 0, 0])),
    (np.array([1, 2, 3]), np.array([0, 1, 2]))
])
def test_l1_proximal_operator(beta, expected):
    npt.assert_array_equal(regs.L1().proximal_operator(beta, 1), expected)
