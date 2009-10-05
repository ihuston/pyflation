"""Configuration file for harness.py
Author: Ian Huston

"""
import numpy as N
import logging
import time
import cosmomodels as c

BASEDIR = "/home/ith/numerics/"
RESULTSDIR = BASEDIR + "results/" + time.strftime("%Y%m%d") + "/"
LOGDIR = BASEDIR + "applogs/"
LOGLEVEL = logging.INFO #Change to desired logging level
POT_FUNC = "hybrid2and4"
NUMSOKS =1025  #Should be power of two + 1
ntheta = 513
YSTART = N.array([25.0, # \phi_0
                0.0, # \dot{\phi_0}
               0.0, # H - leave as 0.0 to let program determine
               1.0, # Re\delta\phi_1
                0.0, # Re\dot{\delta\phi_1}
                1.0, # Im\delta\phi_1
                0.0  # Im\dot{\delta\phi_1}
                ])
#YSTART=None
foclass = c.FOCanonicalTwoStage
cq = 50

FOARGS = {"potential_func": POT_FUNC,
            "ystart": YSTART,
            "cq": cq}
SOARGS = {}

#Standard params
kinit = 1.00e-61
kend = None
deltak = 1.0e-61
