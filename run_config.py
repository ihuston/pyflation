'''
run_config.py Configuration settings for a simulation run
Created on 30 Jun 2010

@author: Ian Huston
'''

import numpy as np
import cosmomodels as c
from configuration import PROGRAM_NAME, LOGLEVEL
from sourceterm import srcequations
import os.path

fixtures = {"msqphisq":        {"potential_func": "msqphisq",
                                "ystart": np.array([18.0, -0.1,0,0,0,0,0])},
            "lambdaphi4":      {"potential_func": "lambdaphi4",
                                "ystart": np.array([25.0, 0,0,0,0,0,0])},
            "hybrid2and4":     {"potential_func": "hybrid2and4",
                                "ystart": np.array([25.0, 0,0,0,0,0,0])},
            "linde":           {"potential_func": "linde",
                                "ystart": np.array([25.0, 0,0,0,0,0,0])},
            "phi2over3":       {"potential_func": "phi2over3",
                                "ystart": np.array([10.0, 0,0,0,0,0,0])},
            "msqphisq_withV0": {"potential_func": "msqphisq_withV0",
                                "ystart": np.array([18.0, 0,0,0,0,0,0])},
            "step_potential":  {"potential_func": "step_potential",
                                "ystart": np.array([18.0, -0.1,0,0,0,0,0])},
            "bump_potential":  {"potential_func": "bump_potential",
                                "ystart": np.array([18.0, -0.1,0,0,0,0,0])}
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
srcclass = srcequations.FullSingleFieldSource
cq = 50

#If sourceterm files already exist should they be overwritten?
overwrite = True



##############################
# DO NOT CHANGE ANYTHING BELOW
# THIS LINE
##############################

from configuration import CODEDIR, RESULTSDIR, LOGDIR, QSUBLOGSDIR, QSUBSCRIPTSDIR, _debug
from configuration import provenancefilename

if not all(map(os.path.isdir, [CODEDIR, RESULTSDIR, LOGDIR, QSUBSCRIPTSDIR, QSUBLOGSDIR])):
    raise IOError("Directory structure is not correct!")

logfile = os.path.join(LOGDIR, "run.log")
provenancefile = os.path.join(LOGDIR, provenancefilename)

#Arguments for first and second order models
pot_func = fx["potential_func"]
ystart = fx["ystart"]

foargs = {"potential_func": pot_func,
            "ystart": ystart,
            "cq": cq,
            "solver": "rkdriver_tsix"}
soargs = {"solver": "rkdriver_tsix"}
 
##############################
# qsub submission values
#
##############################
runname = PROGRAM_NAME[0:4]
qsublogname = os.path.join(QSUBLOGSDIR, "log" )
timelimit = "23:00:00" # Time needed for each array job
taskmin= "1" #starting task id number
taskmax= "100" #finishing task id number
hold_jid_list= "" # List of jobs this task depends on 

templatefile = os.path.join(CODEDIR, "qsub-template.sh")

foscriptname = os.path.join(QSUBSCRIPTSDIR, "fo.qsub")
srcscriptname = os.path.join(QSUBSCRIPTSDIR, "src.qsub")
mrgscriptname = os.path.join(QSUBSCRIPTSDIR, "mrg.qsub")
soscriptname = os.path.join(QSUBSCRIPTSDIR, "so.qsub")
cmbscriptname = os.path.join(QSUBSCRIPTSDIR, "cmb.qsub")

foresults = os.path.join(RESULTSDIR, "fo.hf5")
#Source results will be stored in src-#.hf5
srcstub = os.path.join(RESULTSDIR, "src-")
#This is the pattern that is checked when results are merged
pattern = "src-(\d*).hf5" 

srcresults = os.path.join(RESULTSDIR, "src.hf5")
mrgresults = os.path.join(RESULTSDIR, "mrg.hf5")
soresults = os.path.join(RESULTSDIR, "so.hf5")
cmbresults = os.path.join(RESULTSDIR, "cmb.hf5")

