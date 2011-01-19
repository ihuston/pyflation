"""Configuration file for harness.py
Author: Ian Huston

"""
import logging
import os.path


##############################
# CHANGEABLE VALUES
##############################

# Directory structure

# Calculate base directory as being below the current file
packagedir = os.path.dirname(os.path.abspath(__file__))
CODEDIR = os.path.abspath(os.path.join(packagedir, os.path.pardir))


# Change the names of various directories
#Change to using the base run directory with bin, pyflation, scripts immediately below.
CODEDIRNAME = "." 
RUNDIRNAME = "runs"
RESULTSDIRNAME = "results"
LOGDIRNAME = "applogs"
QSUBSCRIPTSDIRNAME = "qsubscripts"
QSUBLOGSDIRNAME = "qsublogs"

#Directory names computed from current code directory
RUNDIR = CODEDIR
RESULTSDIR = os.path.join(RUNDIR, RESULTSDIRNAME)
LOGDIR = os.path.join(RUNDIR, LOGDIRNAME)
QSUBSCRIPTSDIR = os.path.join(RUNDIR, QSUBSCRIPTSDIRNAME)
QSUBLOGSDIR = os.path.join(RUNDIR, QSUBLOGSDIRNAME)

#Name of provenance file which records the code revisions and results files added
provenancefilename = "provenance.log"


# Compression type to be used with PyTables:
# PyTables stores results in HDF5 files. The compression it uses can be 
# selected here. For maximum compatibility with other HDF5 utilities use "zlib".
# For maximum efficiency in both storage space and recall time use "blosc".
hdf5complib = "blosc"

# The logging level changes how much is saved to logging files. 
# Choose from logging.DEBUG, .INFO, .WARN, .ERROR, .CRITICAL in decreasing order of verbosity
LOGLEVEL = logging.INFO


##################################################
# debug logging control
# 0 for off, 1 for on
# This can be changed using command line arguments
##################################################
_debug = 1

#Program name
PROGRAM_NAME = "Pyflation"
VERSION = "0.1.0"
