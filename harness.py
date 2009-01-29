"""Harness to run multiple simulations at different k
    by Ian huston
    """

from __future__ import division # Get rid of integer division problems, i.e. 1/2=0
import numpy as N
import cosmomodels as c
import scitools.basics as stb
import time
import sys
import logging
import logging.handlers
import sosource
import getopt
import sohelpers
import os

BASEDIR = "/misc/scratch/ith/numerics/"
RESULTSDIR = BASEDIR + "results/" + time.strftime("%Y%m%d") + "/"
LOGDIR = BASEDIR + "applogs/"
LOGLEVEL = logging.INFO #Change to desired logging level
POT_FUNC = "msqphisq"
#YSTART = N.array([25.0, # \phi_0
#                -0.1, # \dot{\phi_0}
#                0.0, # H - leave as 0.0 to let program determine
#                1.0, # Re\delta\phi_1
#                0.0, # Re\dot{\delta\phi_1}
#                1.0, # Im\delta\phi_1
#                0.0  # Im\dot{\delta\phi_1}
#                ])
YSTART=None

FOARGS = {"potential_func": POT_FUNC,
            "ystart": YSTART}
#Get root logger
if __name__ == "__main__":
    harness_logger = logging.getLogger()
    harness_logger.handlers = []
else:
    harness_logger = logging.getLogger(__name__)

def startlogging():
    """Start the logging system to store rotational log based on date."""

    harness_logger.setLevel(LOGLEVEL)
    #Get date for logfile
    date = time.strftime("%Y%m%d")
    #create file handler and set level to debug
    fh = logging.handlers.RotatingFileHandler(filename=LOGDIR + date + ".log", maxBytes=2**20, backupCount=50)
    fh.setLevel(LOGLEVEL)
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
    
def runfomodel(kinit, kend, deltak, filename=None, foargs=None):
    """Execute a FOCanonicalTwoStage model and save results.
    
    A new instance of FOCanonicalTwoStage is created, with the specified arguments.
    The model is run and the results are then saved into a file with the specified filename
    
    Parameters
    ----------
    kinit : float
            Start of sequence of mode numbers to run in model.
    
    kend : float
           End of sequence of mode numbers to run in model.
           
    deltak : float
             Step size in sequence of mode numbers to test.
    
    filename : String, optional
               Name of file to save results to. File will be created in the directory
               specified by `RESULTSDIR` module variable.
               
    foargs : dict, optional
             Dictonary of arguments to be sent to first order class method. 
             If `foargs` contains a key `k` then these mode numbers will be used instead of 
             the sequence generated by `kinit`, `kend` and `deltak`.
    
    Returns
    -------
    filename: String
              Name of the file where results have been saved.
              
    Raises
    ------
    Exception
       Any exception raised during saving of code.
    """
    if foargs is None:
        foargs = {}
    if "k" not in foargs:
        foargs["k"] = stb.seq(kinit, kend, deltak)
    if "solver" not in foargs:
        foargs["solver"] = "rkdriver_withks"
        
    model = c.FOCanonicalTwoStage(**foargs)
    try:
        harness_logger.debug("Starting model run...")
        model.run(saveresults=False)
        harness_logger.debug("Model run finished.")
    except c.ModelError:
        harness_logger.exception("Something went wrong with model, quitting!")
        sys.exit(1)
    if filename is None:
        filename = RESULTSDIR + "fo-" + kinit + "-" + kend + "-" + deltak + "-" + time.strftime("%H%M%S") + ".hf5"
    try:
        harness_logger.debug("Trying to save model data to %s...", filename)
        ensureresultspath(filename)
        model.saveallresults(filename=filename)
        #Success!
        harness_logger.info("Successfully ran and saved simulation in file %s.", filename)
    except Exception:
        harness_logger.exception("IO error, nothing saved!")
    #Destroy model instance to save memory
    harness_logger.debug("Destroying model instance...")
    del model
    
    return filename

def runsomodel(fofile, filename=None, soargs=None):
    """Execute a SOCanonicalThreeStage model and save results.
    
    A new instance of SOCanonicalThreeStage is created, from the specified first order file.
    The model is run and the results are then saved into a file with the specified filename.
    
    Parameters
    ----------
    fofile : String
             Filename of first order file to use in simulation. First order file must contain
             source term and have correct data structure.
    
    filename : String, optional
               Name of file to save results to. File will be created in the directory
               specified by `RESULTSDIR` module variable.
               
    soargs : dict, optional
             Dictonary of arguments to be sent to second order class method. 
    
    Returns
    -------
    filename: String
              Name of the file where results have been saved.
              
    Raises
    ------
    Exception
       Any exception raised during saving of code.
    """
    try:
        fomodel = c.make_wrapper_model(fofile)
    except:
        harness_logger.exception("Error wrapping model file.")
        raise
    if soargs is None:
        soargs = {}
    somodel = c.SOCanonicalThreeStage(fomodel, **soargs)
    try:
        harness_logger.debug("Starting model run...")
        somodel.run(saveresults=False)
        harness_logger.debug("Model run finished.")
    except c.ModelError:
        harness_logger.exception("Something went wrong with model, quitting!")
        sys.exit(1)
    if filename is None:
        filename = RESULTSDIR + "so-" + kinit + "-" + kend + "-" + deltak + "-" + time.strftime("%H%M%S") + ".hf5"
    try:
        harness_logger.debug("Trying to save model data to %s...", filename)
        ensureresultspath(filename)
        somodel.saveallresults(filename=filename)
    except Exception:
        harness_logger.exception("IO error, nothing saved!")
    #Destroy model instance to save memory
#     harness_logger.debug("Destroying model instance...")
#     del somodel
    #Success!
    harness_logger.info("Successfully ran and saved simulation in file %s.", filename)
    return filename

def runfullsourceintegration(modelfile, sourcefile=None):
    """Run source integrand calculation."""
    try:
        m = c.make_wrapper_model(modelfile)
    except:
        harness_logger.exception("Error wrapping model file.")
        raise
    if sourcefile is None:
        sourcefile = RESULTSDIR + "src-" + str(min(m.k)) + "-" + str(max(m.k))
        sourcefile += "-" + str(m.k[1]-m.k[0]) + "-" + time.strftime("%H%M%S") + ".hf5"
    #get source integrand and save to file
    try:
        ensureresultspath(sourcefile)
        filesaved = sosource.getsourceandintegrate(m, sourcefile, intmethod="romb")
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

def dofullrun():
    """Complete full model run of 1st, source and 2nd order calculations."""
    harness_logger.info("Starting full run through...")
    fofile = runfomodel(kinit, kend, deltak, foargs=FOARGS)
    sourcefile = runfullsourceintegration(fofile)
    sohelpers.copy_source_to_fofile(sourcefile, fofile)
    sofile = runsomodel(fofile)
    cfilename = sofile.replace("so", "cmb")
    cfile = sohelpers.combine_results(fofile, sofile, cfilename)
    harness_logger.info("Combined results saved in %s.", cfile)
    return cfile

def main(args):
    """Main function: deal with command line arguments and start calculation as reqd."""

    #Start the logging module
    startlogging()
    
    #Set up arguments
    shortargs = "hf:msta"
    longargs = ["help", "filename=", "fomodel", "somodel", "source", "all"]
    try:                                
        opts, args = getopt.getopt(args, shortargs, longargs)
    except getopt.GetoptError:
        print __doc__ 
        sys.exit(2)
    filename = None
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print __doc__
            sys.exit()
        elif opt in ("-f", "--filename"):
            filename = arg
        elif opt in ("-m", "--fomodel"):
            func = "fomodel"
        elif opt in ("-s", "--somodel"):
            func = "somodel"
        elif opt in ("-t", "--source"):
            func = "source"
        elif opt in ("-a", "--all"):
            func = "all"
    #Standard params
    kinit = 1.00e-62
    kend = 3.00e-62
    deltak = 1.0e-62
    if func == "fomodel":
        try:
            if not filename:
                filename = None
        except AttributeError:
            filename = None 
        #start model run
        runfomodel(kinit, kend, deltak, filename=filename, foargs=FOARGS)
    elif func == "somodel":
        try:
            if not filename:
                raise AttributeError("Need to specify first order file!")
        except AttributeError:
            raise
        #start model run
        runsomodel(fofile=filename)
    elif func == "source":
        try:
            runfullsourceintegration(modelfile=filename)
        except Exception:
            harness_logger.error("Error getting source integral!") 
    elif func == "all":
        try:
            dofullrun(kinit, kend, deltak)
        except Exception:
            harness_logger.error("Error doing full run!")
        
if __name__ == "__main__":
    main(sys.argv[1:])
