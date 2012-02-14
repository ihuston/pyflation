#!/usr/bin/env python
"""source.py - Compute the source term integral given first order results


"""
#Author: Ian Huston
#For license and copyright information see LICENSE.txt which was distributed with this file.


from __future__ import division

import os.path
import numpy as np
import logging
import sys
import optparse

#Try to import run configuration file
try:
    import run_config
except ImportError, e:
    if __name__ == "__main__":
        msg = """Configuration file run_config.py needs to be available."""
        print msg, e
        sys.exit(1)
    else:
        raise

try:
    #Local modules from pyflation package
    from pyflation import helpers, configuration
    from pyflation import cosmomodels as c
    _debug = configuration._debug
    from pyflation.sourceterm import sosource
except ImportError,e:
    if __name__ == "__main__":
        msg = """Pyflation module needs to be available. 
Either run this script from the base directory as bin/source.py or add directory enclosing pyflation package to PYTHONPATH."""
        print msg, e
        sys.exit(1)
    else:
        raise

def runsource(fofile, ninit=0, nfinal=-1, sourcefile=None, 
              ntheta=run_config.ntheta, numsoks=run_config.numsoks, taskarray=None, 
              srcclass=None, overwrite=False):
    """Run parallel source integrand and second order calculation."""
    
    id = taskarray["id"]
    ntasks = (taskarray["max"] -taskarray["min"]) // taskarray["step"] + 1
   
    try:
        m = c.make_wrapper_model(fofile)
    except:
        log.exception("Error wrapping first order file.")
        
    if m.nfields > 1:
        log.exception("""Only single field models can have second order 
        perturbation calculated.""")
        raise c.ModelError
    
    if sourcefile is None:
        sourcefile = run_config.srcstub + str(id) + ".hf5"
    
    if nfinal == -1:
        nfinal = m.tresult.shape[0]
    nfostart = min(m.fotstartindex).astype(int)
    nstar = max(nfostart, ninit)
    totalnrange = len(m.tresult[nstar:nfinal])
    nrange = np.ceil(totalnrange/ntasks)
    
    #Change myninit to match task id
    if id == 1:
        #First task should start at very beginning
        myninit = ninit 
    else:
        #Other tasks start where last one ended
        myninit = nstar + (id-1)*nrange
    #Each task ends after doing it's range of steps
    mynend = nstar + id*nrange
    if mynend > nfinal:
        #Make sure not to go any further than the end
        mynend = nfinal
    if id == taskarray["max"]:
        #Make up any leftover steps
        mynend = nfinal
    
    log.info("Process rank: %d, ninit: %d, nend: %d", id, myninit, mynend)
    if myninit > mynend:
        #No timesteps left so stop
        log.info("Process with rank %d has not timesteps to complete. Quitting!", id)
        return None
    
    #Set source class using run_config
    if srcclass is None:
        srcclass = run_config.srcclass
    
    
    #get source integrand and save to file
    try:
        filesaved = sosource.getsourceandintegrate(m, sourcefile, ninit=myninit, nfinal=mynend,
                                                   ntheta=ntheta, numks=numsoks, srcclass=srcclass,
                                                   overwrite=overwrite)
        log.info("Source term saved as " + filesaved)
    except Exception:
        log.exception("Error getting source term.")
        raise
    #Destroy model instance to save memory
    if _debug:
        log.debug("Destroying model instance to reclaim memory...")
    try:
        del m
    except IOError:
        log.exception("Error closing model file!")
        
        raise
    
    return filesaved
    
    

def main(argv=None):
    """Main function: deal with command line arguments and start calculation as reqd."""
    
    if not argv:
        argv = sys.argv
    
    #Parse command line options
    parser = optparse.OptionParser()
    
    parser.add_option("-f", "--filename", action="store", dest="foresults", 
                      default=run_config.foresults, type="string", 
                      metavar="FILE", help="first order results file, default=%default")
    parser.add_option("--overwrite", action="store_true", dest="overwrite",
                      default=run_config.overwrite,  
                      help="if selected source term files which already exist will be overwritten.")
    
    arraygroup = optparse.OptionGroup(parser, "Task Array Options",
                            "These options specify a task array to work inside. "
                            "The array is the range taskmin:taskmax with step taskstep. "
                            "The current process should be given a taskid in the range specified. "
                            "The default is an array of 1:1, step 1 with id 1.")
    arraygroup.add_option("--taskmin", action="store", dest="taskmin", default=1,
                          type="int", help="start of task array range", metavar="NUM")
    arraygroup.add_option("--taskmax", action="store", dest="taskmax", default=1,
                          type="int", help="end of task array range", metavar="NUM")
    arraygroup.add_option("--taskstep", action="store", dest="taskstep", default=1,
                          type="int", help="step size of task array range", metavar="NUM")
    arraygroup.add_option("--taskid", action="store", dest="taskid", default=1,
                          type="int", help="task id of current process", metavar="NUM")
    parser.add_option_group(arraygroup)
    
    timegroup = optparse.OptionGroup(parser, "Timestep Options",
                     "These options affect which timesteps the source term is calculated for.")
    timegroup.add_option("--tstart", action="store", dest="tstart", default=0,
                         type="int", help="first time step to calculate, default=%default")
    timegroup.add_option("--tend", action="store", dest="tend", default=-1,
                         type="int", help="last time step to calculate, use -1 for the last value, default=%default")
    
    parser.add_option_group(timegroup)
    
    loggroup = optparse.OptionGroup(parser, "Log Options", 
                           "These options affect the verbosity of the log files generated.")
    loggroup.add_option("-q", "--quiet",
                  action="store_const", const=logging.FATAL, dest="loglevel", 
                  help="only print fatal error messages")
    loggroup.add_option("-v", "--verbose",
                  action="store_const", const=logging.INFO, dest="loglevel", 
                  help="print informative messages")
    loggroup.add_option("--debug",
                  action="store_const", const=logging.DEBUG, dest="loglevel", 
                  help="log lots of debugging information",
                  default=run_config.LOGLEVEL)
    loggroup.add_option("--console", action="store_true", dest="console",
                        default=False, help="if selected matches console log level " 
                        "to selected file log level, otherwise only warnings are shown.")
    parser.add_option_group(loggroup)
    
    (options, args) = parser.parse_args(args=argv[1:])
        
            
    #Start the logging module
    if options.console:
        consolelevel = options.loglevel
    else:
        consolelevel = logging.WARN
    
    #Change logger to add task id
    if options.taskmax != options.taskmin:
        log.name = "src-" + str(options.taskid)
        sosource.set_log_name()
        
    logfile = os.path.join(run_config.LOGDIR, "src.log")
    helpers.startlogging(log, logfile, options.loglevel, consolelevel)
    
    if (not _debug) and (options.loglevel == logging.DEBUG):
        log.warn("Debugging information will not be stored due to setting in run_config.")
        
    
    taskarray = dict(min=options.taskmin,
                     max=options.taskmax,
                     step=options.taskstep,
                     id=options.taskid)
    
    if not os.path.isfile(options.foresults):
        raise IOError("First order file %s does not exist! Please run firstorder.py." % options.foresults)
    
    try:
        runsource(fofile=options.foresults, ninit=options.tstart, 
                  nfinal=options.tend, taskarray=taskarray, overwrite=options.overwrite)
    except Exception:
        log.exception("Error getting source integral!")
        return 1
    
    return 0
    
if __name__ == "__main__":
    log = logging.getLogger()
    log.handlers = []
    sys.exit(main())
else:
    log = logging.getLogger("src")
