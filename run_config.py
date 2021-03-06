"""
run_config.py - Configuration settings for a simulation run

The user changeable values in this file are explained below. For each run 
the main options include choice of potential and initial conditions, choice of 
k mode range and selection of first, source and second order python classes.


"""
#Author: Ian Huston
#For license and copyright information see LICENSE.txt which was distributed with this file.


###############################################################################
# DO NOT CHANGE ANYTHING IN THIS SECTION
import numpy as np
import os.path

#Pyflation imports

from pyflation import cosmomodels as c
from pyflation import configuration
from pyflation.sourceterm import srcequations
from pyflation.helpers import getkend
# DO NOT CHANGE ANYTHING ABOVE THIS LINE
###############################################################################

#
# USER CONFIGURABLE VALUES START HERE
#

########### LOGGING ###########################################################
# The logging level changes how much is saved to logging files.
# The default value is set by the configuration module.
LOGLEVEL = configuration.LOGLEVEL 
# To change the default uncomment the LOGLEVEL command below and choose from 
# logging.DEBUG, .INFO, .WARN, .ERROR, .CRITICAL in decreasing order of verbosity.
# LOGLEVEL = logging.INFO
###############################################################################


############ CHOICE OF POTENTIAL AND INITIAL CONDITIONS #######################
# The following dictionary structure contains various present combinations of 
# potentials and initial conditions. To add a new combination just enter it as 
# the next item of the dictionary.
#
# The cq parameter controls how far into the subhorizon stage is the 
# initialisation of each k mode pertubation. Initialisation takes place when
# k/aH = cq. Default value is 50. 

fixtures = {"msqphisq":        {"potential_func": "msqphisq",
                                "pot_params": {"nfields": 1},
                                "nfields": 1,
                                "bgystart": np.array([18.0, -0.1,0]),
                                "cq": 50,
                                "solver": "rkdriver_tsix"},
            "lambdaphi4":      {"potential_func": "lambdaphi4",
                                "pot_params": {"nfields": 1},
                                "nfields": 1,
                                "bgystart": np.array([25.0, 0,0]),
                                "cq": 50,
                                "solver": "rkdriver_tsix"},
            "hybrid2and4":     {"potential_func": "hybrid2and4",
                                "pot_params": {"nfields": 1},
                                "nfields": 1,
                                "bgystart": np.array([25.0, 0,0]),
                                "cq": 50,
                                "solver": "rkdriver_tsix"},
            "linde":           {"potential_func": "linde",
                                "pot_params": {"nfields": 1},
                                "nfields": 1,
                                "bgystart": np.array([25.0, 0,0]),
                                "cq": 50,
                                "solver": "rkdriver_tsix"},
            "phi2over3":       {"potential_func": "phi2over3",
                                "pot_params": {"nfields": 1},
                                "nfields": 1,
                                "bgystart": np.array([10.0, 0,0]),
                                "cq": 50,
                                "solver": "rkdriver_tsix"},
            "msqphisq_withV0": {"potential_func": "msqphisq_withV0",
                                "pot_params": {"nfields": 1},
                                "nfields": 1,
                                "bgystart": np.array([18.0, 0,0]),
                                "cq": 50,
                                "solver": "rkdriver_tsix"},
            "step_potential":  {"potential_func": "step_potential",
                                "pot_params": {"nfields": 1},
                                "nfields": 1,
                                "bgystart": np.array([18.0, -0.1,0]),
                                "cq": 50,
                                "solver": "rkdriver_tsix"},
            "bump_potential":  {"potential_func": "bump_potential",
                                "pot_params": {"nfields": 1},
                                "nfields": 1,
                                "bgystart": np.array([18.0, -0.1,0]),
                                "cq": 50,
                                "solver": "rkdriver_tsix"},
            "hybridquadratic":  {"potential_func": "hybridquadratic",
                                "pot_params": {"nfields": 2},
                                "nfields": 2,
                                "bgystart": np.array([12.0, 1/300.0, 12.0,49/300.0,0]),
                                "cq": 50,
                                "solver": "rkdriver_tsix"},
            "nflation":  {"potential_func": "nflation",
                                "pot_params": {"nfields": 2},
                                "nfields": 2,
                                "bgystart": None, #Defaults to (18,-0.1)/sqrt(nfields)
                                "cq": 50,
                                "solver": "rkdriver_tsix"}, 
            "hybridquartic":  {"potential_func": "hybridquartic",
                                "pot_params": {"nfields": 2},
                                "nfields": 2,
                                "bgystart": np.array([1e-2, 2e-8, 1.63e-9,3.26e-7,0]),
                                "cq": 50,
                                "solver": "rkdriver_tsix"},
            "productexponential":  {"potential_func": "productexponential",
                                "pot_params": {"nfields": 2},
                                "nfields": 2,
                                "bgystart": np.array([18.0, 0.0, 0.001,0,0]),
                                "cq": 50,
                                "solver": "rkdriver_tsix"},
            }

##############################
# CHOOSE FIXTURE HERE
# Choose one of the combinations of potential and initial conditions described
# above by selecting it by name.
foargs = fixtures["msqphisq"]
##############################
###############################################################################

################# WAVEMODE RANGE SELECTION #####################################
# Choose the range for the wavenumber k here. The kinit parameter is the 
# starting value and deltak specifies the difference between consecutive k's.
# The numsoks parameter specifies the number of k values to be calculated during
# the second order perturbation calculation. This should be one plus two to the 
# power of an integer, i.e. 2n+1 for n integer.
# 
# The K_ranges dictionary has pre_defined values for the k range.

K_ranges = { "K1": {"kinit": 0.5e-61, "deltak": 1e-61, "numsoks": 1025},
             "K2": {"kinit": 1.5e-61, "deltak": 3e-61, "numsoks": 1025},
             "K3": {"kinit": 0.25e-60, "deltak": 1e-60, "numsoks": 1025},
             "K4": {"kinit": 0.85e-60, "deltak": 0.4e-60, "numsoks":1025}}

# Pick K_range used here by selecting it from the dictionary above.
K_range = K_ranges["K4"]

# Do not change these lines, which select the initial and delta values from
# the specified range.
kinit = K_range["kinit"]
deltak = K_range["deltak"]
numsoks = K_range["numsoks"]  #Should be power of two + 1

# The end value of the k range is calculated using the getkend function in 
# pyflation.helpers.
kend = getkend(kinit, deltak, numsoks)
###############################################################################


############## MODEL CLASS SELECTION ###########################################
# These options are for advanced users only. 
#
# The driver class used for the first order perturbation calculation can be 
# selected here. It should be accessible from this module, so add imports if
# necessary. The default class is in the pyflation.cosmomodels module.
# The default is c.FOCanonicalTwoStage. 
# To set a fixed a_init value use c.FixedainitTwoStage
foclass = c.FOCanonicalTwoStage

# Here the source term class can be selected. The classes are in the 
# pyflation.sourceterm.srcequations module and the default is 
# srcequations.SelectedkOnlyFullSource. Other options include SlowRollSource, 
# FullSingleFieldSource and SelectedkOnlySlowRollSource.
srcclass = srcequations.SelectedkOnlyFullSource

# The second order perturbation class can also be selected, again from the 
# pyflation.cosmomodels module. The default is c.CanonicalRampedSecondOrder 
# but the unramped version is availabe using c.CanonicalSecondOrder.
soclass = c.CanonicalRampedSecondOrder

# The ntheta parameter controls how finely the [0,pi] range is divided in the 
# integration of the convolution terms. Default is 513.
ntheta = 513



###############################################################################

################ QSUB SUBMISSION OPTIONS ######################################
# These parameters are inserted into the qsub submission scripts
# which are generated and submitted by pyflation-qsubstart.py.
# Please contact your local cluster administrator and consult the qsub man page
# for good values for your local configuration.

timelimit = "23:00:00" # Time needed for each array job
taskmin= "1" #starting task id number
taskmax= "100" #finishing task id number
hold_jid_list= "" # List of jobs this task depends on

############################################################################### 

#
# USER CONFIGURABLE VALUES END HERE
#


###############################################################################
###############################################################################
# DO NOT CHANGE ANYTHING BELOW THIS LINE UNLESS SURE
###############################################################################
###############################################################################

soargs = {"solver": "rkdriver_tsix",
          "nfields": 1, #Only single field models can have second order calced
          "soclass": soclass}

#If sourceterm files already exist should they be overwritten?
overwrite = True

# Calculate code directory as being directory in which cosmomodels.py
# is situated. This should be changed if a more portable system is used.
CODEDIR = os.path.abspath(os.path.dirname(c.__file__))

#Directory names computed from directory in which run_config.py is based.
RUNDIR = os.path.abspath(os.path.dirname(__file__))
RESULTSDIR = os.path.join(RUNDIR, configuration.RESULTSDIRNAME)
LOGDIR = os.path.join(RUNDIR, configuration.LOGDIRNAME)
QSUBSCRIPTSDIR = os.path.join(RUNDIR, configuration.QSUBSCRIPTSDIRNAME)
QSUBLOGSDIR = os.path.join(RUNDIR, configuration.QSUBLOGSDIRNAME)

if not all(map(os.path.isdir, [RESULTSDIR, LOGDIR, QSUBSCRIPTSDIR, QSUBLOGSDIR])):
    raise IOError("Directory structure is not correct!")

# This is the default log file although scripts do write to their own files.
logfile = os.path.join(LOGDIR, "run.log")
 
# qsub script values

runname = "pyfl"
qsublogname = os.path.join(QSUBLOGSDIR, "log" )
templatefilename = "qsub-sh.template"
templatefile = os.path.join(CODEDIR, templatefilename)
foscriptname = os.path.join(QSUBSCRIPTSDIR, "fo.qsub")
srcscriptname = os.path.join(QSUBSCRIPTSDIR, "src.qsub")
src_indivscriptname = os.path.join(QSUBSCRIPTSDIR, "src_individual.qsub")
mrgscriptname = os.path.join(QSUBSCRIPTSDIR, "mrg.qsub")
soscriptname = os.path.join(QSUBSCRIPTSDIR, "so.qsub")
cmbscriptname = os.path.join(QSUBSCRIPTSDIR, "cmb.qsub")

# Results filenames
foresults = os.path.join(RESULTSDIR, "fo.hf5")
#Source results will be stored in src-#.hf5
srcstub = os.path.join(RESULTSDIR, "src-")
#This is the pattern that is checked when results are merged
pattern = "src-(\d*).hf5" 
srcresults = os.path.join(RESULTSDIR, "src.hf5")
mrgresults = os.path.join(RESULTSDIR, "mrg.hf5")
soresults = os.path.join(RESULTSDIR, "so.hf5")
cmbresults = os.path.join(RESULTSDIR, "cmb.hf5")

