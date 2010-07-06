'''source.py - Compute the source term integral given first order results
This should be run as a standalone script.

Created on 6 Jul 2010

@author: Ian Huston
'''

import cosmomodels as c
import run_config
import sosource
import time
import helpers
import os.path
import numpy as N
import srcmerge
import sohelpers
import logging
import sys


def runfullsourceintegration(modelfile, ninit=0, nfinal=-1, sourcefile=None, numsoks=1025, ntheta=513):
    """Run source integrand calculation."""
    try:
        m = c.make_wrapper_model(modelfile)
    except:
        log.exception("Error wrapping model file.")
        raise
    if sourcefile is None:
        sourcefile = run_config.RESULTSDIR + "src-" + m.potential_func + "-" + str(min(m.k)) + "-" + str(max(m.k))
        sourcefile += "-" + str(m.k[1]-m.k[0]) + "-" + time.strftime("%H%M%S") + ".hf5"
    #get source integrand and save to file
    try:
        helpers.ensurepath(sourcefile)
        filesaved = sosource.getsourceandintegrate(m, sourcefile, ninit=ninit, nfinal=nfinal, ntheta=ntheta, numks=numsoks)
        log.info("Source term saved as " + filesaved)
    except Exception:
        log.exception("Error getting source term.")
        raise
    #Destroy model instance to save memory
    log.debug("Destroying model instance to reclaim memory...")
    try:
        del m
    except IOError:
        log.exception("Error closing model file!")
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
        log.exception("Error wrapping model file.")
        if myrank != 0:
            comm.send([{'rank':myrank, 'status':10}], dest=0, tag=10) #Send error msg
        raise
    if sourcefile is None:
        if "fo-" in modelfile:
            srcstub = os.path.splitext(os.path.basename(modelfile))[0].replace("fo", "src")
        else:
            srcstub = "-".join(["src", m.potential_func, str(min(m.k)), str(max(m.k)), str(m.k[1]-m.k[0]), time.strftime("%Y%m%d")])
        sourcefile = run_config.RESULTSDIR + srcstub + "/src-part-" + str(myrank) + ".hf5"
    if myrank == 0:
        #Check sourcefile directory exists:
        helpers.ensurepath(os.path.dirname(sourcefile))
        log.info("Source file path is %s." %sourcefile)
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
        log.info("Process rank: %d, ninit: %d, nend: %d", myrank, myninit, mynend)
        
        #get source integrand and save to file
        try:
            filesaved = sosource.getsourceandintegrate(m, sourcefile, ninit=myninit, nfinal=mynend,
                                                       ntheta=ntheta, numks=numsoks)
            log.info("Source term saved as " + filesaved)
        except Exception:
            log.exception("Error getting source term.")
            comm.send([{'rank':myrank, 'status':10}], dest=0, tag=10) #Tag=10 signals an error
            raise
        #Destroy model instance to save memory
        log.debug("Destroying model instance to reclaim memory...")
        try:
            del m
        except IOError:
            log.exception("Error closing model file!")
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
                log.error("Error in subprocess %d!", status.source)
                raise IOError("Error in subprocess %d!" % status.source)
        log.info("All processes finished! Starting merger...")
        srcdir = os.path.dirname(sourcefile)
        newsrcfile = os.path.dirname(srcdir) + os.sep + srcstub + ".hf5" 
        srcmergefile = srcmerge.mergefiles(newfile=newsrcfile, dirname=srcdir)
        log.info("Merger complete. File saved as %s.", srcmergefile)
        #Start combination of first order and source files
        log.info("Starting to combine first order and source files.")
        foandsrcfile = sohelpers.combine_source_and_fofile(srcmergefile, modelfile)
        log.info("Combination complete, saved in %s.", foandsrcfile)
        log.info("Starting second order run...")
        sofile = runsomodel(foandsrcfile, soargs=soargs)
        log.info("Second order run complete. Starting to combine first and second order results.")
        cfilename = sofile.replace("so", "cmb")
        cfile = sohelpers.combine_results(foandsrcfile, sofile, cfilename)
        log.info("Combined results saved in %s.", cfile)
        return cfile
    
def main():
    , "source", "all", "debug", "kinit=", "kend=", "deltak=", "parallelsrc", "begin=", "end=", "numsoks=", "ntheta="
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
    
    elif func == "source":
        log.info("-----------Source integral run requested------------------")
        log.info("Parameters: modelfile=%s, ntheta=%s", str(filename), str(ntheta))
        try:
            runfullsourceintegration(modelfile=filename, ninit=ninit, nfinal=nfinal, ntheta=ntheta, numsoks=numsoks)
        except Exception:
            log.exception("Error getting source integral!")
    elif func == "parallelsrc":
        try:
            try:
                runparallelintegration(modelfile=filename, ninit=ninit, nfinal=nfinal, ntheta=ntheta, numsoks=numsoks, soargs=soargs)
            except ImportError:
                log.exception("Parallel module not available!")
                sys.exit(1)
        except Exception:
            log.exception("Error getting source integral in parallel!")
    
if __name__ == "__main__":
    log = logging.getLogger(__name__)
    log.handlers = []
    main(sys.argv[1:])
else:
    log = logging.getLogger(__name__)
