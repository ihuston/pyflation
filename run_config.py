'''
run_config.py Configuration settings for a simulation run
Created on 30 Jun 2010

@author: Ian Huston
'''

import numpy as N
import cosmomodels as c
import configuration
import os.path
import helpers

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
# kinit, deltak values
# Add range to K_ranges to change values
##############################
K_ranges = { "K1": {"kinit": 0.5e-61, "deltak": 1e-61, "numsoks": 1025},
             "K2": {"kinit": 1.5e-61, "deltak": 3e-61, "numsoks": 1025},
             "K3": {"kinit": 0.25e-60, "deltak": 1e-60, "numsoks": 1025}}

#Pick K_range used
K_range = K_ranges["K1"]

#Do not change these values
kinit = K_range["kinit"]
deltak = K_range["deltak"]
numsoks = K_range["numsoks"]  #Should be power of two + 1

def getkend(kinit, deltak, numsoks):
    """Correct kend value given the values of kinit, deltak and numsoks.
    """
    #Change from numsoks-1 to numsoks to include extra point when deltak!=kinit
    return 2*((numsoks)*deltak + kinit)
    
kend = getkend(kinit, deltak, numsoks)

##############################
# IMPORTANT VALUES 
# DO NOT CHANGE UNLESS SURE
##############################


ntheta = 513
foclass = c.FOCanonicalTwoStage
cq = 50

##############################
# qsub submission values
#
##############################


##############################
# DO NOT CHANGE ANYTHING BELOW
# THIS LINE
##############################

#Directory names computed from current code directory
RUNDIR = os.path.dirname(configuration.CODEDIR)
RESULTSDIR = os.path.join(RUNDIR, configuration.RESULTSDIRNAME)
LOGDIR = os.path.join(RUNDIR, configuration.LOGDIRNAME)
QSUBSCRIPTSDIR = os.path.join(RUNDIR, configuration.QSUBSCRIPTSDIRNAME)
QSUBLOGSDIR = os.path.join(RUNDIR, configuration.QSUBLOGSDIRNAME)

if not all(map(os.path.isdir, [RESULTSDIR, LOGDIR, QSUBSCRIPTSDIR, QSUBLOGSDIR])):
    raise IOError("Directory structure is not correct!")

#Dictionary of values for qsub scripts
qsubdict = {}

#Arguments for first and second order models
pot_func = fx["potential_func"]
ystart = fx["ystart"]

foargs = {"potential_func": pot_func,
            "ystart": ystart,
            "cq": cq}
soargs = {}
 

