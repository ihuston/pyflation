"""Configuration file for harness.py
Author: Ian Huston

"""
import numpy as N
import logging
import time
import cosmomodels as c
import os.path

######################################
#IMPORTANT
#CHANGE TO BRANCH YOU WANT TO RUN FROM
######################################
BZRBRANCH = "trunk" 
######################################


fixtures = {"msqphisq":        {"potential_func": "msqphisq",
                                "ystart": N.array([18.0, -0.1,0,0,0,0,0])},
            "lambdaphi4":      {"potential_func": "lambdaphi4",
                                "ystart": N.array([25.0, 0,0,0,0,0,0])},
            "hybrid2and4":     {"potential_func": "hybrid2and4",
                                "ystart": N.array([25.0, 0,0,0,0,0,0])},
            "linde":           {"potential_func": "linde",
                                "ystart": N.array([25.0, 0,0,0,0,0,0])},
            "phi2over3":       {"potential_func": "phi2over3",
                                "ystart": N.array([10.0, 0,0,0,0,0,0])},
            "msqphisq_withV0": {"potential_func": "msqphisq_withV0",
                                "ystart": N.array([18.0, 0,0,0,0,0,0])}
            }

##############################
# CHOOSE FIXTURE HERE
fx = fixtures["msqphisq"]
##############################

##############################
# SOME OTHER CHANGEABLE VALUES
##############################
BASEDIR = os.path.join(os.path.expandvars("$HOME"), "pyflation")
if not os.path.isdir(BASEDIR):
    raise IOError("Base directory %s does not exist" % BASEDIR)

 
CODEDIR = os.path.join(BASEDIR, "code", BZRBRANCH)
RESULTSDIR = os.path.join(BASEDIR, "results", time.strftime("%Y%m%d"))
LOGDIR = os.path.join(BASEDIR, "applogs")
LOGLEVEL = logging.INFO #Change to desired logging level
QSUBSCRIPTSDIR = os.path.join(BASEDIR, "qsubscripts")
QSUBLOGSDIR = os.path.join(BASEDIR, "qsublogs")

##############################
# IMPORTANT VALUES 
# DO NOT CHANGE UNLESS SURE
##############################
NUMSOKS =1025  #Should be power of two + 1
ntheta = 513
foclass = c.FOCanonicalTwoStage
cq = 50

##############################
# DO NOT CHANGE ANYTHING BELOW
# THIS LINE
##############################
POT_FUNC = fx["potential_func"]
YSTART = fx["ystart"]
FOARGS = {"potential_func": POT_FUNC,
            "ystart": YSTART,
            "cq": cq}
SOARGS = {}

##############################
#Old params for compatibility
kinit = 1.00e-61
kend = None
deltak = 1.0e-61
##############################
