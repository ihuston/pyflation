*********
Pyflation
*********

Introduction
============

Pyflation is a Python package for calculating cosmological perturbations during
inflationary expansion of the universe. 

Pyflation was created by Ian Huston (http://www.ianhuston.net) while at 
Queen Mary, University of London (http://www.qmul.ac.uk).

Once installed the modules in the pyflation package can be used to run 
simulations of different scalar field models of the early universe.

The main classes are contained in the cosmomodels module and include 
simulations of background fields and first order and second order perturbations.
The sourceterm package contains modules required for the computation of the 
term required for the evolution of second order perturbations.
The analysis package contains routines to calculate the first order 
First and Second order perturbations can be calculated for single field models 
but only first order perturbations can be evolved for multi-field models.

Alongside the Python package, the bin directory contains Python scripts which 
can run first and second order simulations.
A helper script called "pyflation_qsubstart.py" sets up a full second order run (including 
background, first order and source calculations)
to be used on queueing system which contains the "qsub" executable (e.g. a Rocks 
cluster).

Installation of the code is described in the INSTALL.txt file. The pyflation
package can be installed as a normal Python package to the site-packages 
directory or each run of the code can be self-contained with code, results and 
logs all contained in a run directory.

The "pyflation_newrun.py" script creates a new run directory and populates it with the 
code and sub-directories which are required.
In particular the file "provenence.log" in the "applogs" directory contains 
information about the version of the code and system libraries at the time of 
the creation of the run.

More information about Pyflation is available at the website 
http://pyflation.ianhuston.net. Online documentation is available at 
http://pyflation.ianhuston.net/docs. 

For copyright and license information please see the LICENSE.txt file. For 
installation information see the INSTALL.txt file. For usage information see 
the USAGE.txt file.

Acknowledgements
================

This author of this software is supported by the Science & Technology Facilities 
Council under grant ST/G002150/1.

Author: Ian Huston

