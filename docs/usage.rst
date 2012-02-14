*****
Usage
*****

Basic usage of the pyflation package
====================================

The pyflation package can be used independently of the scripts which run full
first and second order calculations. If you have installed the package using
the setup.py install command then it should be available by doing::

    > import pyflation

in an interactive Python session. If you have not installed the package then 
make sure that the path where the pyflation directory is located has been 
added to the PYTHONPATH variable.

The most important module in the package is cosmomodels. This contains the 
classes which drive first and second order evolution. To run a first order
simulation with the default settings first import the cosmomodels module::

    > from pyflation import cosmomodels as c

Here the module is aliased as c for convenience. Then create a new first order
model using the FOCanonicalTwoStage class::

    > m = c.FOCanonicalTwoStage()

You can inspect the attributes and methods of m in the usual way by using 
dir(m) or tab completion in the iPython environment.

To begin the first order run use::

    > m.run()

The results are stored in m.yresult in raw form for all time steps. 
The value of the first order perturbations for all time steps are also labelled 
as m.dp1.  

Basic Usage of Scripts
======================

Assuming you have followed the instructions in INSTALL.txt to install the program
either system-wide, locally, or in a contained directory, you can create a new
run directory using the pyflation-newrun.py script.

If you have used the setup.py install command, the script should be somewhere
on your PATH. If not you will need to prefix the following commands with the 
path to the pyflation-newrun.py script.
To create a new run in the directory $HOME/pyflation-runs and call the run
mynewrun run the pyflation-newrun.py script with the following options::

    $ pyflation-newrun.py -d $HOME/pyflation-runs -n mynewrun

The script should create the directory (and any parent ones needed) and set up 
the file structure inside. Change to the run directory and look inside. The 
directories applogs, qsublogs, qsubscripts and results are created automatically 
and the run_config.py file is put inside the run directory. If you want the code 
to be copied in to the run directory please see the advanced options of the 
pyflation-newrun.py script.

The run_config.py file inside the run directory contains user changeable settings.
In particular the choice of potential and k range to be used in the scripts is
made here. The python classes used for the different stages of the evolution
can also be changed in this file.

The script files are descriptively named. To begin a first order perturbation
calculation use pyflation-firstorder.py. For each script the --help option will
show all the options available. Simple operation uses the defaults in 
run_config.py for example::

    $ pyflation-firstorder.py

will run the first order code and store the result in the file specified in 
run_config.py, usually results/fo.hf5.