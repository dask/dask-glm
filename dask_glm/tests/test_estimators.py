import pytest


def test_warns():
    with pytest.warns(FutureWarning) as w:
        import dask_glm.estimators  # noqa

    assert len(w)
    assert 'dask-ml' in str(w[-1])
