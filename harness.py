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

RESULTSDIR = "/misc/scratch/ith/numerics/results/"
LOGDIR = "/misc/scratch/ith/numerics/applogs/"
LOGLEVEL = logging.DEBUG #Change to desired logging level
#Get root logger
if __name__ == "__main__":
    harness_logger = logging.getLogger()
else:
    harness_logger = logging.getLogger(__name__)

def startlogging():
    """Start the logging system to store rotational log based on date."""
    harness_logger.setLevel(LOGLEVEL)
    #Get date for logfile
    date = time.strftime("%Y%m%d")
    #create file handler and set level to debug
    fh = logging.handlers.RotatingFileHandler(filename=LOGDIR + date + ".log", maxBytes=2**20, backupCount=50)
    fh.setLevel(logging.INFO)
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

def runmodel(kinit, kend, deltak, filename=None):
    """Run program from kinit to kend using deltak"""
    #Start the logging module
    startlogging()
    
    model = c.FONewCanonicalTwoStage(solver="rkdriver_withks", k=stb.seq(kinit, kend, deltak))
    #model = c.SOCanonicalThreeStage(solver="rkdriver_withks")
    try:
        harness_logger.debug("Starting model run...")
        model.run(saveresults=False)
        harness_logger.debug("Model run finished.")
    except c.ModelError:
        harness_logger.exception("Something went wrong with model, quitting!")
        sys.exit(1)
    if filename is None:
        filename = RESULTSDIR + "batchrun" + time.strftime("%Y%m%d%H%M%S") + ".hf5"
    try:
        harness_logger.debug("Trying to save model data to %s...", filename)
        model.saveallresults(filename=filename)
    except Exception:
        harness_logger.exception("IO error, nothing saved!")
    #Destroy model instance to save memory
    harness_logger.debug("Destroying model instance...")
    del model
    #Success!
    harness_logger.info("Successfully ran and saved simulation in file %s.", filename)
    
    return filename

def runsomodel(fofile, filename=None):
    """Run program from kinit to kend using deltak"""
    #Start the logging module
    startlogging()
    try:
        fomodel = c.FOModelWrapper(fofile)
    except:
        harness_logger.exception("Error wrapping model file.")
        raise
    
    somodel = c.SOCanonicalThreeStage(fomodel)
    try:
        harness_logger.debug("Starting model run...")
        somodel.run(saveresults=False)
        harness_logger.debug("Model run finished.")
    except c.ModelError:
        harness_logger.exception("Something went wrong with model, quitting!")
        sys.exit(1)
    if filename is None:
        filename = RESULTSDIR + "batchrun" + time.strftime("%Y%m%d%H%M%S") + ".hf5"
    try:
        harness_logger.debug("Trying to save model data to %s...", filename)
        somodel.saveallresults(filename=filename)
    except Exception:
        harness_logger.exception("IO error, nothing saved!")
    #Destroy model instance to save memory
#     harness_logger.debug("Destroying model instance...")
#     del somodel
    #Success!
    harness_logger.info("Successfully ran and saved simulation in file %s.", filename)
    
    return filename

def runsourceint(modelfile, sourcefile=None):
    """Run source integrand calculation."""
    startlogging()
    try:
        m = c.FOModelWrapper(modelfile)
    except:
        harness_logger.exception("Error wrapping model file.")
        raise
    #get source integrand and save to file
    try:
        filesaved = sosource.getsourceintegrand(m, sourcefile)
        harness_logger.info("Integrand saved as " + filesaved)
    except Exception:
        harness_logger.exception("Error getting source integrand.")
        raise
    #Destroy model instance to save memory
    harness_logger.debug("Destroying model instance to reclaim memory...")
    try:
        del m
    except IOError:
        harness_logger.exception("Error closing model file!")
        raise
       
    return filesaved

def runfullsourceintegration(modelfile, sourcefile=None):
    """Run source integrand calculation."""
    startlogging()
    try:
        m = c.FOModelWrapper(modelfile)
    except:
        harness_logger.exception("Error wrapping model file.")
        raise
    #get source integrand and save to file
    try:
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

def integratesource(modelfile, sourcefile=None):
    """Run source integration calculation."""
    startlogging()
    try:
        m = c.FOModelWrapper(modelfile)
    except:
        harness_logger.exception("Error wrapping model file.")
        raise
    #get source integrand and save to file
    try:
        filesaved = sosource.getsource(m, sourcefile)
        harness_logger.info("Integrand saved as " + filesaved)
    except Exception:
        harness_logger.exception("Error getting source integrand.")
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
    #arguments
#     import psyco
# #     psyco.log()
# #     psyco.profile()
#     psyco.full()
    
    shortargs = "hf:msib"
    longargs = ["help", "filename=", "model", "somodel", "integrand", "both"]
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
        elif opt in ("-m", "--model"):
            func = "model"
        elif opt in ("-s", "--somodel"):
            func = "somodel"
        elif opt in ("-i", "--integrand"):
            func = "integrand"
        elif opt in ("-b", "--both"):
            func = "both"
    
    if func == "model":
        #need to add these to command line options
        kinit = 1.00e-61
        kend = 1.025e-58
        deltak = 1.0e-61
        try:
            if not filename:
                filename = None
        except AttributeError:
            filename = None 
        #start model run
        runmodel(kinit, kend, deltak, filename=filename)
    elif func == "somodel":
        try:
            if not filename:
                raise AttributeError("Need to specify first order file!")
        except AttributeError:
            raise
        #start model run
        runsomodel(fofile=filename)
    elif func == "integrand":
        try:
            runsourceint(filename)
        except Exception:
            harness_logger.error("Error getting source integral!") 
    elif func == "both":
        try:
            runfullsourceintegration(modelfile=filename)
        except Exception:
            harness_logger.error("Error getting source integral!") 

if __name__ == "__main__":
    main(sys.argv[1:])
