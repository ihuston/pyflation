# -*- coding: utf-8 -*-
"""
firstorder.py - Run a first order simulation
Author: Ian Huston

This program will run a first order Pyflation simulation as a straight
through run.  
See cosmomodels.py for the specification for each type of model. 
Default configuration can be changed in run_config.py.


"""

from __future__ import division # Get rid of integer division problems, i.e. 1/2=0

import cosmomodels as c
import sys
import os
import run_config
import optparse
import helpers
import logging


 
def runfomodel(filename=None, foargs=None, foclass=None):
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
             the sequence generated by `kinit`, `kend` and `deltak` as specified in configuration.py.
     
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
    if not issubclass(foclass, c.TwoStageModel):
        raise ValueError("Must use TwoStageModel class for first order run!")
    
    #Create model instance
    model = foclass(**foargs)
    
    try:
        log.debug("Starting model run...")
        model.run(saveresults=False)
        log.debug("Model run finished.")
    except c.ModelError:
        log.exception("Something went wrong with model, quitting!")
        sys.exit(1)
        
    #Save data
    try:
        log.debug("Trying to save model data to %s...", filename)
        helpers.ensurepath(filename)
        model.saveallresults(filename=filename)
        #Success!
        log.info("Successfully ran and saved simulation in file %s.", filename)
    except Exception:
        log.exception("IO error, nothing saved!")
        
    #Destroy model instance to save memory
    log.debug("Destroying model instance...")
    del model
    
    return filename


def main(argv=None):
    """Main function: deal with command line arguments and start calculation as reqd."""
    
    if not argv:
        argv = sys.argv
    
    #Parse command line options
    parser = optparse.OptionParser()
    
    parser.add_option("-f", "--filename", action="store", dest="foresults", 
                      default=run_config.foresults, type="string", 
                      metavar="FILE", help="file to store results")
    
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
                  help="print lots of debugging information",
                  default=run_config.LOGLEVEL)
    parser.add_option_group(loggroup)
    
    (options, args) = parser.parse_args(args=argv[1:])
        
        
    #Start the logging module
    helpers.startlogging(log, run_config.logfile, options.loglevel)
    
    
    log.info("-----------First order run requested------------------")
    
    if os.path.isfile(options.foresults):
        raise IOError("First order results file already exists!") 
      
    foargs = run_config.foargs.copy()
    #Get k variable for first order run
    foargs["k"] = helpers.seq(run_config.kinit, run_config.kend, run_config.deltak)
    
    #Start first order run
    runfomodel(filename=options.foresults, foargs=foargs)
    
    return 0
    
        
if __name__ == "__main__":
    log = logging.getLogger(__name__)
    log.handlers = []
    sys.exit(main())
else:
    log = logging.getLogger(__name__)
