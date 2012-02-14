#!/usr/bin/env python
"""secondorder.py - Run a second order simulation

"""
#Author: Ian Huston
#For license and copyright information see LICENSE.txt which was distributed with this file.


from __future__ import division

import logging
import sys
import os.path
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
except ImportError,e:
    if __name__ == "__main__":
        msg = """Pyflation module needs to be available. 
Either run this script from the base directory as bin/secondorder.py or add directory enclosing pyflation package to PYTHONPATH."""
        print msg, e
        sys.exit(1)
    else:
        raise



def runsomodel(mrgfile, filename=None, soargs=None, sodriver=None):
    """Execute a three stage driver model and save results.
    
    A new instance of sodriver class (default is SOCanonicalThreeStage) is created, from the specified first order file.
    The model is run and the results are then saved into a file with the specified filename.
    
    Parameters
    ----------
    mrgfile : String
             Filename of merged first order and source file to use in simulation. 
    
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
    if soargs.get("nfields", 1) > 1:
        log.exception("""Only single field models can have second order 
        perturbation calculated.""")
        raise c.ModelError
    try:
        fomodel = c.make_wrapper_model(mrgfile)
    except:
        log.exception("Error wrapping model file.")
        raise
    if soargs is None:
        soargs = run_config.soargs
    #Create second order model instance
    if sodriver is None:
        sodriver = c.SOCanonicalThreeStage
    somodel = sodriver(fomodel, **soargs)
    try:
        if _debug:
            log.debug("Starting model run...")
        somodel.run(saveresults=False)
        if _debug:
            log.debug("Model run finished.")
    except c.ModelError:
        log.exception("Something went wrong with model, quitting!")
        sys.exit(1)
    if filename is None:
        filename = run_config.soresults
    try:
        if _debug:
            log.debug("Trying to save model data to %s...", filename)
        helpers.ensurepath(filename)
        somodel.saveallresults(filename=filename, 
                             hdf5complevel=configuration.hdf5complevel,
                             hdf5complib=configuration.hdf5complib)
        #Success!
        log.info("Successfully ran and saved simulation in file %s.", filename)
    except Exception:
        log.exception("IO error, nothing saved!")
    #Destroy model instance to save memory
    if _debug:
        log.debug("Destroying model instance...")
    del somodel
    
    return filename


def main(argv=None):
    """Main function: deal with command line arguments and start calculation as reqd."""
    
    if not argv:
        argv = sys.argv
    
    #Parse command line options
    parser = optparse.OptionParser()
    
    parser.add_option("-f", "--filename", action="store", dest="mrgresults", 
                      default=run_config.mrgresults, type="string", 
                      metavar="FILE", help="merged first order and source results file, default=%default")
    
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
        
    logfile = os.path.join(run_config.LOGDIR, "so.log")
    helpers.startlogging(log, logfile, options.loglevel, consolelevel)
    
    if (not _debug) and (options.loglevel == logging.DEBUG):
        log.warn("Debugging information will not be stored due to setting in run_config.")
    
    if not os.path.isfile(options.mrgresults):
        raise IOError("Merged results file %s does not exist!" % options.foresults)
    
    try:
        log.info("-----------Second order run requested------------------")
        runsomodel(mrgfile=options.mrgresults, soargs=run_config.soargs)
    except Exception:
        log.exception("Error getting second order results!")
        return 1
    
    return 0
        
      

    
    
if __name__ == "__main__":
    log = logging.getLogger()
    log.name = "so"
    log.handlers = []
    sys.exit(main())
else:
    log = logging.getLogger("so")