import cupy
from dask_glm.utils import sigmoid

def cupy_make_y(X, beta=cupy.array([1.5, -3])):
    n, p = X.shape
    z0 = X.dot(beta)
    y = cupy.random.random(z0.shape) < sigmoid(z0)
    return y
