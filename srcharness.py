"""srcharness.py - Tool to run source integration of first order results
Author: Ian Huston

Given the filename of a first order model's results, this tool will calculate
the source term integral for the second order equation of motion. See cosmomodels.py
and harness.py for first order models and a program to run them.

Usage
-----
python srcharness.py -f filename [options]

Arguments
---------
-h, --help:                 Print this help text.
-f file, --filename file:   First order file to use
-t, --source:               Calculate source term (default)
-b num, --begin num:        Begin sourceterm calculation at timestep num
-e num, --end num:          End sourceterm calculation at timestep num (not inclusive)

"""

from __future__ import division # Get rid of integer division problems, i.e. 1/2=0
import numpy as N
import cosmomodels as c
import time
import sys
import logging.handlers
import sosource
import getopt
import sohelpers
import os
import configuration


#Get root logger
if __name__ == "__main__":
    harness_logger = logging.getLogger()
    harness_logger.handlers = []
else:
    harness_logger = logging.getLogger(__name__)

def startlogging():
    """Start the logging system to store rotational log based on date."""

    harness_logger.setLevel(configuration.LOGLEVEL)
    #Get date for logfile
    date = time.strftime("%Y%m%d")
    #create file handler and set level to debug
    fh = logging.handlers.RotatingFileHandler(filename=configuration.LOGDIR + date + ".log", maxBytes=2**20, backupCount=50)
    fh.setLevel(configuration.LOGLEVEL)
    #create console handler and set level to error
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    #create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    #add formatter to fh
    fh.setFormatter(formatter)
    #add formatter to ch
    ch.setFormatter(formatter)
    #add fh to logger
    harness_logger.addHandler(fh)
    #add ch to logger
    harness_logger.addHandler(ch)
    harness_logger.debug("Logging started.")

def ensureresultspath(path):
    """Check that the path for results directory exists and create it if not."""
    #Does path exist?
    if not os.path.isdir(os.path.dirname(path)):
        try:
            os.mkdir(os.path.dirname(path))
        except OSError:
            harness_logger.error("Error creating results directory!")

def runfullsourceintegration(modelfile, ninit=0, nfinal=-1, sourcefile=None):
    """Run source integrand calculation."""
    try:
        m = c.make_wrapper_model(modelfile)
    except:
        harness_logger.exception("Error wrapping model file.")
        raise
    if sourcefile is None:
        sourcefile = configuration.RESULTSDIR + "src-" + m.potential_func + "-" + str(min(m.k)) + "-" + str(max(m.k))
        sourcefile += "-" + str(m.k[1]-m.k[0]) + "-" + time.strftime("%H%M%S") + ".hf5"
    #get source integrand and save to file
    try:
        ensureresultspath(sourcefile)
        filesaved = sosource.getsourceandintegrate(m, sourcefile, ninit=ninit, nfinal=nfinal)
        harness_logger.info("Source term saved as " + filesaved)
    except Exception:
        harness_logger.exception("Error getting source term.")
        raise
    #Destroy model instance to save memory
    harness_logger.debug("Destroying model instance to reclaim memory...")
    try:
        del m
    except IOError:
        harness_logger.exception("Error closing model file!")
        raise
       
    return filesaved

def main(args):
    """Main function: deal with command line arguments and start calculation as reqd."""

    #Start the logging module
    startlogging()
    
    #Set up arguments
    shortargs = "hf:tpb:e:"
    longargs = ["help", "filename=", "source", "parallel", "begin=", "end="]
    try:                                
        opts, args = getopt.getopt(args, shortargs, longargs)
    except getopt.GetoptError:
        print __doc__ 
        sys.exit(2)
    filename = None
    ninit = 0
    nfinal = -1
    func = None
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print __doc__
            sys.exit()
        elif opt in ("-f", "--filename"):
            filename = arg
        elif opt in ("-t", "--source"):
            func = "source"
        elif opt in ("-p", "--parallel"):
            func = "parallel"
        elif opt in ("-b", "--begin"):
            ninit = int(arg)
        elif opt in ("-e", "--end"):
            nfinal = int(arg)
    if func == "source":
        try:
            runfullsourceintegration(modelfile=filename, ninit=ninit, nfinal=nfinal)
        except Exception:
            harness_logger.exception("Error getting source integral!")
    elif func == "parallel":
        try:
            runparallelintegration(modelfile=filename, ninit=ninit, nfinal=nfinal)
        except Exception:
            harness_logger.exception("Error getting source integral in parallel!")
    else:
        print __doc__
        sys.exit()
     
if __name__ == "__main__":
    main(sys.argv[1:])
