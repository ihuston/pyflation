"""Configuration file for harness.py

Author: Ian Huston
For license and copyright information see LICENSE.txt which was distributed with this file.


"""
import logging

##################################################
# debug logging control
# 0 for off, 1 for on
##################################################
_debug = 1

#This is the default log level which can be overridden in run_config.
# The logging level changes how much is saved to logging files. 
# Choose from logging.DEBUG, .INFO, .WARN, .ERROR, .CRITICAL in decreasing order of verbosity
LOGLEVEL = logging.INFO

# Directory structure
# Change the names of various directories
#Change to using the base run directory with bin, pyflation, scripts immediately below.
CODEDIRNAME = "." 
RUNDIRNAME = "runs"
RESULTSDIRNAME = "results"
LOGDIRNAME = "applogs"
QSUBSCRIPTSDIRNAME = "qsubscripts"
QSUBLOGSDIRNAME = "qsublogs"
RUNCONFIGTEMPLATE = "run_config.template"

#Name of provenance file which records the code revisions and results files added
provenancefilename = "provenance.log"

# Compression type to be used with PyTables:
# PyTables stores results in HDF5 files. The compression it uses can be 
# selected here. For maximum compatibility with other HDF5 utilities use "zlib".
# For maximum efficiency in both storage space and recall time use "blosc".
hdf5complib = "blosc"
hdf5complevel = 2






