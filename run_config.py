'''
run_config.py Configuration settings for a simulation run
Created on 30 Jun 2010

@author: Ian Huston
'''

import numpy as N
import os.path

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

