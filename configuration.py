"""Configuration file for harness.py
Author: Ian Huston

"""
import numpy as N
import logging
import time
import cosmomodels as c
import os.path


##############################
# CHANGEABLE VALUES
##############################

# Directory structure

# Name of base directory which everything else is below
BASEDIRNAME = "pyflation"


# Calculate base directory as being below $HOME
CODEDIR = os.path.dirname(os.path.abspath(__file__))

if os.path.basename(os.path.dirname(CODEDIR)) == BASEDIRNAME:
    BASEDIR = os.path.dirname(CODEDIR)
elif os.path.basename(os.path.dirname(os.path.dirname(CODEDIR))) == BASEDIRNAME:
    BASEDIR = os.path.dirname(os.path.dirname(CODEDIR))
elif os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(CODEDIR)))) == BASEDIRNAME:
    BASEDIR = os.path.dirname(os.path.dirname(os.path.dirname(CODEDIR)))
else:
    raise IOError("Base directory cannot be found!")


# Change the names of various directories
CODEDIRNAME = "code"
RUNDIRNAME = "runs"
RESULTSDIRNAME = "results"
LOGDIRNAME = "applogs"
QSUBSCRIPTSDIRNAME = "qsubscripts"
QSUBLOGSDIRNAME = "qsublogs"


# The logging level changes how much is saved to logging files. 
# Choose from logging.DEBUG, .INFO, .WARN, .ERROR, .CRITICAL in decreasing order of verbosity
LOGLEVEL = logging.INFO 

#Program name
PROGRAM_NAME = "Pyflation"
VERSION = "0.2"
