# -*- coding: utf-8 -*-
"""
Harness to run multiple simulations at different k
Author: Ian Huston

This program will run Cosmomodels simulations in different stage or as a straight
through run. For a full through or first order run no filename is required, but can be
specified if desired. For a source or second order run you need to give the filename of
a first order model's results. This tool will calculate the source term integral for the
second order equation of motion but does not automatically combine the source term into
the first order results file in preparation of a second order run. 
See cosmomodels.py for the specification for each type of model. 
Configuration is done in hconfig.py.

Usage
-----
python harness.py [-f filename] [options]

Arguments
---------
-h, --help:                 Print this help text.
-f file, --filename file:   First order file to use
-a, --all:                  Run all stages of computation
-m, --fomodel:              Run first order model
-p, --parallelsrc:          Calculate source term in parallel (need first order filename)
-t, --source:               Calculate source term (single process) (need first order filename)
-s, --somodel:              Run second order model (need combined foandsrc filename)
-b num, --begin num:        Begin sourceterm calculation at timestep num
-e num, --end num:          End sourceterm calculation at timestep num (not inclusive)
-d, --debug:                Change logging level to debug, dramatic increase in volume.
--kinit num:                Initial k mode to use in k range
--deltak num:               Difference between two k modes in k range
--kend num:                 Final k mode to use in k range (Can be calculated from kinit, deltak)


"""

from __future__ import division # Get rid of integer division problems, i.e. 1/2=0
import numpy as N
import cosmomodels as c
from scitools.basics import seq 
import time
import sys
import logging
import logging.handlers
import sosource
import getopt
import sohelpers
import os
import hconfig
import srcmerge 
from helpers import ensurepath


def startlogging(loglevel=hconfig.LOGLEVEL):
    """Start the logging system to store rotational log based on date."""

    harness_logger.setLevel(loglevel)
    #Get date for logfile
    date = time.strftime("%Y%m%d")
    #create file handler and set level to debug
    fh = logging.handlers.RotatingFileHandler(filename=hconfig.LOGDIR + date + ".log", maxBytes=2**20, backupCount=50)
    fh.setLevel(loglevel)
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
    harness_logger.debug("Logging started at level %d", loglevel)



def checkkend(kinit, kend, deltak, numsoks):
    """Check whether kend has correct value for second order run.
    
    If it is None, then set it to a value that will satisfy requirements. 
    If it is not the correct value then log a complaint and return it anyway.
    """
    if not kend:
        #Change from numsoks-1 to numsoks to include extra point when deltak!=kinit
        kend = 2*((numsoks)*deltak + kinit)
        harness_logger.info("Set kend to %s.", str(kend))
    elif kend < 2*((numsoks)*deltak + kinit):
        harness_logger.info("Requested k range will not satisfy condition for second order run!")
    return kend
    
def runfomodel(filename=None, foargs=None, foclass=hconfig.foclass):
    """Execute a TwoStageModel from cosmomodels and save results.
    
    A new instance of foclass is created, with the specified arguments.
    The model is run and the results are then saved into a file with the specified filename
    
    Parameters
    ----------
    filename: String, optional
               Name of file to save results to. File will be created in the directory
               specified by `RESULTSDIR` module variable.
               
    foargs: dict, optional
             Dictonary of arguments to be sent to first order class method. 
             If `foargs` contains a key `k` then these mode numbers will be used instead of 
             the sequence generated by `kinit`, `kend` and `deltak` as specified in hconfig.py.
     
    foclass: class object, optional
             Class to use as model. Should be a subclass of cosmomodels.TwoStageModel.
             Defaults to class given in configuration file.
    
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
        kinit, kend, deltak, numsoks = hconfig.kinit, hconfig.kend, hconfig.deltak, hconfig.NUMSOKS
        kend = checkkend(kinit, kend, deltak, numsoks)
        foargs["k"] = seq(kinit, kend, deltak)
    if "solver" not in foargs:
        foargs["solver"] = "rkdriver_withks"
    if "potential_func" not in foargs:
        foargs["potential_func"] = hconfig.POT_FUNC
    if "ystart" not in foargs:
        foargs["ystart"] = hconfig.YSTART
    if filename is None:
        kinit, kend, deltak = foargs["k"][0], foargs["k"][-1], foargs["k"][1]-foargs["k"][0]
        filename = hconfig.RESULTSDIR + "fo-" + foclass.__name__ + "-" + foargs["potential_func"] + "-" + str(kinit) + "-" + str(kend) + "-" + str(deltak) + "-" + time.strftime("%H%M%S") + ".hf5"
    if not issubclass(foclass, c.TwoStageModel):
        raise ValueError("Must use TwoStageModel class for first order run!")
    
    model = foclass(**foargs)
    try:
        harness_logger.debug("Starting model run...")
        model.run(saveresults=False)
        harness_logger.debug("Model run finished.")
    except c.ModelError:
        harness_logger.exception("Something went wrong with model, quitting!")
        sys.exit(1)
    try:
        harness_logger.debug("Trying to save model data to %s...", filename)
        ensurepath(filename)
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
        kinit, kend, deltak = somodel.k[0], somodel.k[-1], somodel.k[1]-somodel.k[0]
        filename = hconfig.RESULTSDIR + "so-" + somodel.potential_func + "-" + str(kinit) + "-" + str(kend) + "-" + str(deltak) + ".hf5"
    try:
        harness_logger.debug("Trying to save model data to %s...", filename)
        ensurepath(filename)
        somodel.saveallresults(filename=filename)
    except Exception:
        harness_logger.exception("IO error, nothing saved!")
    #Destroy model instance to save memory
#     harness_logger.debug("Destroying model instance...")
#     del somodel
    #Success!
    harness_logger.info("Successfully ran and saved simulation in file %s.", filename)
    return filename
    
def runfullsourceintegration(modelfile, ninit=0, nfinal=-1, sourcefile=None, numsoks=1025, ntheta=513):
    """Run source integrand calculation."""
    try:
        m = c.make_wrapper_model(modelfile)
    except:
        harness_logger.exception("Error wrapping model file.")
        raise
    if sourcefile is None:
        sourcefile = hconfig.RESULTSDIR + "src-" + m.potential_func + "-" + str(min(m.k)) + "-" + str(max(m.k))
        sourcefile += "-" + str(m.k[1]-m.k[0]) + "-" + time.strftime("%H%M%S") + ".hf5"
    #get source integrand and save to file
    try:
        ensurepath(sourcefile)
        filesaved = sosource.getsourceandintegrate(m, sourcefile, ninit=ninit, nfinal=nfinal, ntheta=ntheta, numks=numsoks)
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

def runparallelintegration(modelfile, ninit=0, nfinal=None, sourcefile=None, ntheta=513, numsoks=1025, soargs=None):
    """Run parallel source integrand and second order calculation."""
    try:
        from mpi4py import MPI
    except ImportError:
        raise

    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
    nprocs = comm.Get_size() - 1 #Don't include host
    status = 0
    try:
        m = c.make_wrapper_model(modelfile)
    except:
        harness_logger.exception("Error wrapping model file.")
        if myrank != 0:
            comm.send([{'rank':myrank, 'status':10}], dest=0, tag=10) #Send error msg
        raise
    if sourcefile is None:
        if "fo-" in modelfile:
            srcstub = os.path.splitext(os.path.basename(modelfile))[0].replace("fo", "src")
        else:
            srcstub = "-".join(["src", m.potential_func, str(min(m.k)), str(max(m.k)), str(m.k[1]-m.k[0]), time.strftime("%Y%m%d")])
        sourcefile = hconfig.RESULTSDIR + srcstub + "/src-part-" + str(myrank) + ".hf5"
    if myrank == 0:
        #Check sourcefile directory exists:
        ensurepath(os.path.dirname(sourcefile))
    if myrank != 0:
        #Do not include host node in execution
        if nfinal == -1:
            nfinal = m.tresult.shape[0]
        nfostart = min(m.fotstartindex).astype(int)
        nstar = max(nfostart, ninit)
        totalnrange = len(m.tresult[nstar:nfinal])
        nrange = N.ceil(totalnrange/nprocs)
        if myrank == 1:
            myninit = ninit
        else:
            myninit = nstar + (myrank-1)*nrange
        mynend = nstar + myrank*nrange
        harness_logger.info("Process rank: %d, ninit: %d, nend: %d", myrank, myninit, mynend)
        
        #get source integrand and save to file
        try:
            filesaved = sosource.getsourceandintegrate(m, sourcefile, ninit=myninit, nfinal=mynend,
                                                       ntheta=ntheta, numks=numsoks)
            harness_logger.info("Source term saved as " + filesaved)
        except Exception:
            harness_logger.exception("Error getting source term.")
            comm.send([{'rank':myrank, 'status':10}], dest=0, tag=10) #Tag=10 signals an error
            raise
        #Destroy model instance to save memory
        harness_logger.debug("Destroying model instance to reclaim memory...")
        try:
            del m
        except IOError:
            harness_logger.exception("Error closing model file!")
            comm.send([{'rank':myrank, 'status':10}], dest=0, tag=10) #Tag=10 signals an error
            raise
        comm.send([{'rank':myrank, 'status':0}], dest=0, tag=0) #Tag=0 signals success
        return filesaved
    else:
        #Get rid of model object
        del m
        process_list = range(1, nprocs+1)
        status_list = []
        while len(status_list) < len(process_list):
            status = MPI.Status()
            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            status_list.append(data[0])
            if status.tag > 0:
                harness_logger.error("Error in subprocess %d!", status.source)
                raise IOError("Error in subprocess %d!" % status.source)
        harness_logger.info("All processes finished! Starting merger...")
        srcdir = os.path.dirname(sourcefile)
        newsrcfile = os.path.dirname(srcdir) + os.sep + srcstub + ".hf5" 
        srcmergefile = srcmerge.mergefiles(newfile=newsrcfile, dirname=srcdir)
        harness_logger.info("Merger complete. File saved as %s.", srcmergefile)
        #Start combination of first order and source files
        harness_logger.info("Starting to combine first order and source files.")
        foandsrcfile = sohelpers.combine_source_and_fofile(srcmergefile, modelfile)
        harness_logger.info("Combination complete, saved in %s.", foandsrcfile)
        harness_logger.info("Starting second order run...")
        sofile = runsomodel(foandsrcfile, soargs=soargs)
        harness_logger.info("Second order run complete. Starting to combine first and second order results.")
        cfilename = sofile.replace("so", "cmb")
        cfile = sohelpers.combine_results(foandsrcfile, sofile, cfilename)
        harness_logger.info("Combined results saved in %s.", cfile)
        return cfile
          
def dofullrun():
    """Complete full model run of 1st, source and 2nd order calculations."""
    harness_logger.info("---------------------Starting full run through...--------------------")
    fofile = runfomodel(foargs=hconfig.FOARGS)
    sourcefile = runfullsourceintegration(fofile)
    foandsrcfile = sohelpers.combine_source_and_fofile(sourcefile, fofile)
    sofile = runsomodel(foandsrcfile)
    cfilename = sofile.replace("so", "cmb")
    cfile = sohelpers.combine_results(fofile, sofile, cfilename)
    harness_logger.info("Combined results saved in %s.", cfile)
    harness_logger.info("---------------- Full run finished! ---------------------")
    return cfile

def main(args):
    """Main function: deal with command line arguments and start calculation as reqd."""

    #Set up arguments
    shortargs = "hf:mstadpb:e:"
    longargs = ["help", "filename=", "fomodel", "somodel", "source", "all", "debug", "kinit=", "kend=", "deltak=", "parallelsrc", "begin=", "end=", "numsoks=", "ntheta="]
    try:                                
        opts, args = getopt.getopt(args, shortargs, longargs)
    except getopt.GetoptError:
        print __doc__ 
        sys.exit(2)
    filename = None
    func = None
    kinit = kend = deltak = None
    ninit = 0
    nfinal = -1
    numsoks = hconfig.NUMSOKS
    ntheta = hconfig.ntheta
    loglevel = hconfig.LOGLEVEL
    foargs = getattr(hconfig, "FOARGS", {})
    soargs = getattr(hconfig, "SOARGS", {})
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
        elif opt in ("-d", "--debug"):
            loglevel = logging.DEBUG
        elif opt in ("--kinit",):
            kinit = float(arg)
        elif opt in ("--kend",):
            kend = float(arg)
        elif opt in ("--deltak",):
            deltak = float(arg)
        elif opt in ("-p", "--parallelsrc"):
            func = "parallelsrc"
        elif opt in ("-b", "--begin"):
            ninit = int(arg)
        elif opt in ("-e", "--end"):
            nfinal = int(arg)
        elif opt in ("--numsoks",):
            numsoks = int(arg)
        elif opt in ("--ntheta",):
            ntheta = int(arg)
    #Start the logging module
    startlogging(loglevel)
    
    if func == "fomodel":
        harness_logger.info("-----------First order run requested------------------")
        try:
            if not filename:
                filename = None
        except AttributeError:
            filename = None 
        #start model run
        if not (kinit and deltak):
            kinit, deltak, kend = hconfig.kinit, hconfig.deltak, hconfig.kend
        if not kend:
            kend = 2*((numsoks-1)*deltak + kinit)
            harness_logger.info("Set kend to %s.", str(kend))
        elif kend < 2*((numsoks-1)*deltak + kinit):
            harness_logger.info("Requested k range will not satisfy condition for second order run!")
        
        foargs["k"] = seq(kinit, kend, deltak)
        runfomodel(filename=filename, foargs=foargs)
    elif func == "somodel":
        harness_logger.info("-----------Second order run requested------------------")
        try:
            if not filename:
                raise AttributeError("Need to specify first order file!")
        except AttributeError:
            harness_logger.exception("Error starting second order model!")
        #start model run
        runsomodel(fofile=filename, soargs=soargs)
    elif func == "source":
        harness_logger.info("-----------Source integral run requested------------------")
        harness_logger.info("Parameters: modelfile=%s, ntheta=%s", str(filename), str(ntheta))
        try:
            runfullsourceintegration(modelfile=filename, ninit=ninit, nfinal=nfinal, ntheta=ntheta, numsoks=numsoks)
        except Exception:
            harness_logger.exception("Error getting source integral!")
    elif func == "parallelsrc":
        try:
            try:
                runparallelintegration(modelfile=filename, ninit=ninit, nfinal=nfinal, ntheta=ntheta, numsoks=numsoks, soargs=soargs)
            except ImportError:
                harness_logger.exception("Parallel module not available!")
                sys.exit(1)
        except Exception:
            harness_logger.exception("Error getting source integral in parallel!")
    elif func == "all":
        try:
            dofullrun()
        except Exception:
            harness_logger.exception("Error doing full run!")
    else:
        print __doc__
        sys.exit()
        
if __name__ == "__main__":
    harness_logger = logging.getLogger()
    harness_logger.handlers = []
    main(sys.argv[1:])
else:
    harness_logger = logging.getLogger(__name__)
