#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pyflation_firstorder.py - Run a first order Pyflation simulation

This program will run a first order Pyflation simulation as a straight
through run.  
See cosmomodels.py for the specification for each type of model. 
Default configuration can be changed in run_config.py.


"""
#Author: Ian Huston
#For license and copyright information see LICENSE.txt which was distributed with this file.


from __future__ import division # Get rid of integer division problems, i.e. 1/2=0

import sys
import os
import optparse
import logging

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
Either run this script from the base directory as bin/firstorder.py or add directory enclosing pyflation package to PYTHONPATH."""
        print msg, e
        sys.exit(1)
    else:
        raise


 
def runfomodel(filename=None, foargs=None, foclass=None):
    """Execute a FOCanonicalTwoStage driver from cosmomodels and save results.
    
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
             the sequence generated by `kinit`, `kend` and `deltak` as specified in configuration.py.
     
    foclass: class object, optional
             Class to use as model. Should be a subclass of cosmomodels.FOCanonicalTwoStage.
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
    #Check whether foargs is specified and use default if not. 
    if foargs is None:
        foargs = run_config.foargs.copy()
        
    #Check whether needed array k is in foargs otherwise use values from run_config
    if "k" not in foargs:
        kinit, kend, deltak= (run_config.kinit, run_config.kend, run_config.deltak)
        foargs["k"] = helpers.seq(kinit, kend, deltak)
    
    if filename is None:
        filename = run_config.foresults
    
    #Check foclass is specified and is suitable
    if not foclass:
        foclass = run_config.foclass
    if not issubclass(foclass, c.FOCanonicalTwoStage):
        raise ValueError("Must use FOCanonicalTwoStage class for first order run!")
    
    #Create model instance
    model = foclass(**foargs)
    
    log.debug("Path for model data file is %s.", filename)
    try:
        helpers.ensurepath(filename)
    except IOError:
        log.exception("Error creating save file directory.")
        return 1
    
    #Create saving arguments
    saveargs = dict(filename=filename, 
                    hdf5complevel=configuration.hdf5complevel,
                    hdf5complib=configuration.hdf5complib)
    
    try:
        log.debug("Starting model run...")
        model.run(saveresults=True, saveargs=saveargs)
        log.debug("Model run finished.")
    except c.ModelError:
        log.exception("Something went wrong with model, quitting!")
        return 1
    except IOError:
        log.exception("IO error, nothing saved!")
        return 1    
    #Save data
    
    
    
    #Success!
    log.info("Successfully ran and saved simulation in file %s.", filename)

        
    #Destroy model instance to save memory
    log.debug("Destroying model instance...")
    try:
        del model
    except:
        pass
    
    return 0


def main(argv=None):
    """Deal with command line arguments and start calculation as required.
    
    Parameters
    ----------
    
    argv: list
          List of command line arguments in the same form as sys.argv, 
          i.e. argv[0] is the name of command being run.
          
    Returns
    -------
    int: Return value equivalent to shell return code, 0 corresponds to successful execution,
         a value not equal to 0 corresponds to an error.
    
    """
    
    if not argv:
        argv = sys.argv
    
    #Parse command line options
    parser = optparse.OptionParser()
    
    parser.add_option("-f", "--filename", action="store", dest="foresults", 
                      default=run_config.foresults, type="string", 
                      metavar="FILE", help="file to store results")
    
    parser.add_option("-p", "--potential", action="store", dest="fixture", 
                      default="", type="string", 
                      metavar="POTENTIAL", 
                      help="the potential to use, choose from the fixtures dict in run_config.py")
    
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
    helpers.startlogging(log, run_config.logfile, options.loglevel, consolelevel)
    
    
    log.info("-----------First order run requested------------------")
    
    if os.path.isfile(options.foresults):
        raise IOError("First order results file already exists!") 
    
    if options.fixture != "":
        try:
            foargs = run_config.fixtures[options.fixture]
            log.info("Potential %s selected." % foargs["potential_func"])
        except ValueError:
            log.exception("Fixture doesn't exist in run_config.py!")
    else:
        foargs = run_config.foargs.copy()
        
    #Get k variable for first order run
    foargs["k"] = helpers.seq(run_config.kinit, run_config.kend, run_config.deltak)
    
    #Start first order run
    try:
        retval = runfomodel(filename=options.foresults, foargs=foargs)
    except Exception:
        log.exception("Something went wrong during first order run!")
        return 1
    if retval != 0:
        log.error("Model run ended with error!")
        return retval
    
    log.info("----------First order run finished-------------------")
    return 0
    
        
if __name__ == "__main__":
    log = logging.getLogger()
    log.name = "fo"
    log.handlers = []
    sys.exit(main())
else:
    log = logging.getLogger("fo")
