""" setup.py - Script to install package using distutils

Author: Ian Huston
For license and copyright information see LICENSE.txt which was distributed with this file.


For help options run:
$ python setup.py help
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
from pyflation import __version__ as pyflation_version

###############
VERSION = pyflation_version


ext_modules = [Extension("pyflation.sourceterm.srccython", ["pyflation/sourceterm/srccython.pyx"],
               include_dirs=[numpy.get_include()]),
               #
               Extension("pyflation.romberg", ["pyflation/romberg.pyx"], 
               include_dirs=[numpy.get_include()]),
               #
               ]


setup_args = dict(name='pyflation',
                  version=VERSION,
                  author='Ian Huston',
                  author_email='ian.huston@gmail.com',
                  url='http://pyflation.ianhuston.net',
                  packages=['pyflation', 'pyflation.sourceterm',
                            'pyflation.analysis'],
                  scripts=['bin/pyflation_firstorder.py', 
                           'bin/pyflation_source.py', 
                           'bin/pyflation_secondorder.py', 
                           'bin/pyflation_combine.py',
                           'bin/pyflation_srcmerge.py', 
                           'bin/pyflation_qsubstart.py',
                           'bin/pyflation_newrun.py'],
                  package_data={'pyflation': ['qsub-sh.template', 
                                              'run_config.template']},
                  cmdclass = {'build_ext': build_ext},
                  ext_modules = ext_modules,
                  license="Modified BSD license",
                  description="""Pyflation is a Python package for calculating 
cosmological perturbations during an inflationary expansion of the universe.""",
                  long_description=open('README.txt').read())

if __name__ == "__main__":
    setup(**setup_args)
