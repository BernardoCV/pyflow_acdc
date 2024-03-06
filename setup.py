# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:54:16 2024

@author: BernardoCastro
"""

from setuptools import setup

setup(name='pyflow_acdc',
      version='0.1',
      description='A python-based tool for the design and analysis of hybrid AC/DC grids',
      url='https://github.com/BernardoCV/pyflow_acdc',
      author='Bernardo Castro Valerio',
      author_email='bernardo.castro@upc.edu',
      license='BSD',
      packages=['pyflow_acdc'],
      python_requires='>=3.8',
      install_requires=['numpy',
                        'pandas',
                        'networkx',
                        'matplotlib',
                        'scipy',
                        'prettytable'],
      extras_require={"OPF":["pymo", "ipopt"]
          
          }
      )
