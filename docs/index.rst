.. pyflation documentation master file, created by
   sphinx-quickstart on Tue Feb 14 13:57:23 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   
*********
Pyflation 
*********
Pyflation is a Python package for calculating cosmological perturbations during an inflationary 
expansion of the universe.

Pyflation was created by `Ian Huston <http://www.ianhuston.net>`_ while at `Queen Mary, University of London 
<http://www.qmul.ac.uk>`_.

Introduction
============

Pyflation is a Python package for calculating cosmological perturbations during
inflationary expansion of the universe. 

Once installed the modules in the :doc:`pyflation` can be used to run 
simulations of different scalar field models of the early universe.

The main classes are contained in the :py:mod:`pyflation.cosmomodels` module and include 
simulations of background fields and first order and second order perturbations.
The :doc:`pyflation.sourceterm` contains modules required for the computation of the 
term required for the evolution of second order perturbations.
The :doc:`pyflation.analysis` contains routines to calculate the first order 
First and Second order perturbations can be calculated for single field models 
but only first order perturbations can be evolved for multi-field models.

Alongside the Python package, the bin directory contains Python scripts which 
can run first and second order simulations.
A helper script called "pyflation_qsubstart.py" sets up a full second order run (including 
background, first order and source calculations)
to be used on queueing system which contains the "qsub" executable (e.g. a Rocks 
cluster).

Installation of the code is described on the :doc:`installation` page. The pyflation
package can be installed as a normal Python package to the site-packages 
directory or each run of the code can be self-contained with code, results and 
logs all contained in a run directory.

The "pyflation_newrun.py" script creates a new run directory and populates it with the 
code and sub-directories which are required.
In particular the file "provenence.log" in the "applogs" directory contains 
information about the version of the code and system libraries at the time of 
the creation of the run.

More information about Pyflation is available at the website 
http://pyflation.ianhuston.net. 


Contents
========
.. toctree::
   :maxdepth: 2
   
   installation
   usage
   scripts
   changelog
   license

Package and Module Reference
============================
.. toctree::
   :maxdepth: 2

   pyflation
   pyflation.analysis
   pyflation.sourceterm


Acknowledgements
================

This author of this software is supported by the Science & Technology Facilities 
Council under grant ST/G002150/1.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

