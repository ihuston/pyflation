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


setup_args = dict(name='Pyflation',
                  version=VERSION,
                  author='Ian Huston',
                  author_email='ian.huston@gmail.com',
                  url='http://www.maths.qmul.ac.uk/~ith/pyflation',
                  description='Cosmological Inflation in Python',
                  packages=['pyflation', 'pyflation.sourceterm',
                            'pyflation.solutions'],
                  scripts=['bin/firstorder.py', 'bin/source.py', 
                           'bin/secondorder.py', 'bin/combine.py',
                           'bin/srcmerge.py', 'bin/start.py',
                           'bin/newrun.py'],
                  package_data={'pyflation': ['qsub-sh.template', 
                                              'run_config.template']},
                  cmdclass = {'build_ext': build_ext},
                  ext_modules = ext_modules)

if __name__ == "__main__":
    setup(**setup_args)
