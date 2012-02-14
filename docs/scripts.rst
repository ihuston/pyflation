*******
Scripts
*******

Description of available scripts
================================

* The pyflation-newrun.py script creates a new run directory and the needed
  directories and files inside it. Options include the ability to copy the 
  code into the new run directory to keep it contained.

* The pyflation-firstorder.py script runs background and first order perturbation
  simulations and saves the results, by default in results/fo.hf5.

* The pyflation-source.py script runs a full source term calculation using the
  first order results. This may take a long time especially if it is not executed
  in parallel. The options for this script include the location of the first
  order results file and simple assignment of task number in a parallel (or multiple
  serial) environment.

* The pyflation-srcmerge.py script will merge any separate source files, for 
  example from a parallel run, into one main source file. Rudimentary checking
  of the existence of each time step is completed. One of the options to this 
  script is to further merge the first order and source term results together.
  The new merged file is used as the input for the second order calculation.

* The pyflation-secondorder.py script uses the source term and first order
  results from the merged file to calculate the second order perturbation solution.

* The pyflation-combine.py script combines all the results from the first, 
  source and second order runs into one final results file. In this file the
  timesteps from the second order run are used so only half the results of the 
  first and source term calculations are stored. For this reason this file is more 
  for ease of analysis then long term data storage.

* The pyflation-qsubstart.py script is used when the qsub queueing command is
  available. See the section "Using Qsub" below for more information. 

Using Qsub 
==========

If you are running Pyflation in a cluster environment or somewhere with the
qsub queueing system available, there is an extra pyflation-qsubstart.py script
which will be useful. When you create a new run the file qsub-sh.template is 
created which holds a very general outline of a qsub run script. Change any 
of the options in the preliminary section to meet your needs. The length of runs
and the number of task units can be set in run_config.py.
When you run::

    $ pyflation-qsubstart.py

a new batch of qsub jobs will be created and submitted, running a full second
order calculation. Please note that the full calculation make take some 
considerable time depending on the available computing power. The scripts which
are submitted using qsub are available in the directory qsub-scripts. An 
additional src_individual.qsub script is useful if one of the tasks in the long
source term calculation fails for some reason. To restart just one task with
the same options as before use::

    $ qsub src_individual.qsub N

where N is the number of the task to restart.