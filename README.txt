Pyflation is a Python package for calculating cosmological perturbations during
inflationary expansion of the universe.

Once installed the modules in the pyflation package can be used to run 
simulations of different scalar field models of the early universe.

The main classes are contained in the cosmomodels module and include 
simulations of background fields and first order and second order perturbations.
The sourceterm package contains modules required for the computation of the 
term required for the evolution of second order perturbations.
The solutions package contains analytical solutions of simple functions which 
can be compared with the numerical output to check the accuracy.

Alongside the Python package, the bin directory contains Python scripts which 
can run first and second order simulations.
A helper script called "start.py" sets up a full second order run (including 
background, first order and source calculations)
to be used on queueing system which contains the "qsub" executable (e.g. a Rocks 
cluster).

Each run of the code is designed to be self-contained with code, results and 
logs all contained in a run directory.
The "newrun.py" script creates a new run directory and populates it with the 
code and sub-directories which are required.
In particular the file "provenence.log" in the "applogs" directory contains 
information about the version of the code and system libraries at the time of 
the creation of the run.

More information about Pyflation is available at the website 
http://www.maths.qmul.ac.uk/~ith/pyflation. 
