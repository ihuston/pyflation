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

# Calculate base directory as being below $HOME
CODEDIR = os.path.dirname(os.path.abspath(__file__))


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
