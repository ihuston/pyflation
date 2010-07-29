from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import configuration

ext_modules = [Extension("srccython", ["srccython.pyx"], 
               extra_compile_args=["-g"], 
               extra_link_args=["-g"],
               include_dirs=[numpy.get_include()]),
               #
               Extension("romberg", ["romberg.pyx"], 
               extra_compile_args=["-g"], 
               extra_link_args=["-g"],
               include_dirs=[numpy.get_include()]),
               #
               #Extension("cythontesting", ["cythontesting.pyx"], 
               #extra_compile_args=["-g"], 
               #extra_link_args=["-g"],
               #include_dirs=[numpy.get_include()])
               ]


setup_args = dict(name=configuration.PROGRAM_NAME,
                  cmdclass = {'build_ext': build_ext},
                  ext_modules = ext_modules)

if __name__ == "__main__":
    setup(**setup_args)
