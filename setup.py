#!/usr/bin/env python

from os.path import exists
from setuptools import setup
import versioneer


setup(name='dask-glm',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Generalized Linear Models with Dask',
      url='http://github.com/dask/dask-glm/',
      maintainer='Matthew Rocklin',
      maintainer_email='mrocklin@gmail.com',
      license='BSD',
      keywords='dask,glm',
      classifiers=[
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
      ],
      packages=['dask_glm'],
      long_description=(open('README.rst').read() if exists('README.rst')
                        else ''),
      install_requires=list(open('requirements.txt').read().strip().split('\n')),
      extras_require={
          'docs': [
              'jupyter',
              'nbsphinx',
              'notebook',
              'numpydoc',
              'sphinx',
              'sphinx_rtd_theme',
          ]
      },
      zip_safe=False)
