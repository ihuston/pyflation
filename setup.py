from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
from pyflation import __version__ as pyflation_version

###############
VERSION = pyflation_version


ext_modules = [Extension("pyflation.sourceterm.srccython", ["pyflation/sourceterm/srccython.pyx"], 
               extra_compile_args=["-g"], 
               extra_link_args=["-g"],
               include_dirs=[numpy.get_include()]),
               #
               Extension("pyflation.romberg", ["pyflation/romberg.pyx"], 
               extra_compile_args=["-g"], 
               extra_link_args=["-g"],
               include_dirs=[numpy.get_include()]),
               #
               ]


setup_args = dict(name='pyflation',
                  version=VERSION,
                  author='Ian Huston',
                  author_email='ian.huston@gmail.com',
                  url='http://www.maths.qmul.ac.uk/~ith/pyflation',
                  packages=['pyflation', 'pyflation.sourceterm'],
                  scripts=['bin/pyflation-firstorder.py', 
                           'bin/pyflation-source.py', 
                           'bin/pyflation-secondorder.py', 
                           'bin/pyflation-combine.py',
                           'bin/pyflation-srcmerge.py', 
                           'bin/pyflation-qsubstart.py',
                           'bin/pyflation-newrun.py'],
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
