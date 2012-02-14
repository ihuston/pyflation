""" setup.py - Script to install package using distutils

For help options run:
$ python setup.py help

"""
#Author: Ian Huston
#For license and copyright information see LICENSE.txt which was distributed with this file.



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
                  long_description=open('README.txt').read(),
                  classifiers=["Intended Audience :: Science/Research",
                               "License :: OSI Approved :: BSD License",
                               "Operating System :: OS Independent",
                               "Programming Language :: Python",
                               "Programming Language :: Python :: 2.6",
                               "Programming Language :: Cython",
                               "Topic :: Scientific/Engineering :: Astronomy",
                               "Topic :: Scientific/Engineering :: Physics"
                               ],
                  requires=["numpy (>= 1.3)",
                                    "scipy (>= 0.7.1)",
                                    "Cython (>= 0.12.1)",
                                    "tables (>= 2.2)"],
                  #Include dependency on matplotlib for cosmographs
                  #extras_require=["matplotlib >= 1.0.1"]
                  )

if __name__ == "__main__":
    setup(**setup_args)
