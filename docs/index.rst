.. dask-glm documentation master file, created by
   sphinx-quickstart on Mon May  1 22:00:08 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Dask-glm
========

*Dask-glm is a library for fitting Generalized Linear Models on large datasets*

Dask-glm builds on the `dask`_ project to fit `GLM`_'s on datasets in parallel.
It provides the optimizers and regularizers used by libraries like `dask-ml`_,
which builds scikit-learn-style APIs on top of those components.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   examples
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _dask: http://dask.pydata.org/en/latest/
.. _GLM: https://en.wikipedia.org/wiki/Generalized_linear_model
.. _scikit-learn: http://scikit-learn.org/
.. _dask-ml: http://dask-ml.readthedocs.org/
