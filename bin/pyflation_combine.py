#!/usr/bin/env python
"""pyflation_combine.py - Combine second order, first order and source results in one file.

"""
#Author: Ian Huston
#For license and copyright information see LICENSE.txt which was distributed with this file.


from __future__ import division

import tables
import numpy as np
import os
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
    _debug = configuration._debug
except ImportError,e:
    if __name__ == "__main__":
        msg = """Pyflation module needs to be available. 
Either run this script from the base directory as bin/combine.py or add directory enclosing pyflation package to PYTHONPATH."""
        print msg, e
        sys.exit(1)
    else:
        raise


def combine_results(fofile, sofile, newfile=None):
    """Combine the first and second order results from given files, and save in newfile."""
    if not newfile:
        newfile = run_config.cmbresults
        log.info("Filename set to " + newfile)
        
    if os.path.isdir(os.path.dirname(newfile)):
        if os.path.isfile(newfile):
            raise IOError("File already exists!")
        else:
            log.debug("File does not exist, using write mode.")
            filemode = "w" #Writing to new file
    else:
        raise IOError("Directory 'results' does not exist")
    #Add compression
    filters = tables.Filters(complevel=configuration.hdf5complevel, 
                             complib=configuration.hdf5complib)
    try:
        sf = tables.openFile(sofile, "r")
        ff = tables.openFile(fofile, "r")
        nf = tables.openFile(newfile, filemode, filters=filters)
    except IOError:
        log.exception("Error opening files!")
        raise
    try:
        #Create groups required
        comgrp = nf.createGroup(nf.root, "results", "Combined first and second order results")
        log.debug("Results group created in combined file.")
        #Store bg results:
        bggrp = nf.copyNode(ff.root.bgresults, nf.root, recursive=True)
        log.debug("Bg results copied.")
        #Save results
        oldyshape = list(sf.root.results.yresult[0:0].shape) #2nd order shape
        oldyshape[1] += ff.root.results.yresult.shape[1] #add number of 1st order vars
        yresarr = nf.createEArray(comgrp, "yresult", tables.Float64Atom(), oldyshape, filters=filters, chunkshape=(10,7,10))
        log.debug("New yresult array with shape %s created.", str(oldyshape))
        #Copy other important arrays
        karr = nf.copyNode(sf.root.results.k, comgrp)
        foparams = nf.copyNode(ff.root.results.parameters, comgrp, newname="foparameters")
        soparams = nf.copyNode(sf.root.results.parameters, comgrp, newname="soparameters")
        #Copy parameters and change classname for compatibility
        params = nf.copyNode(sf.root.results.parameters, comgrp, newname="parameters")
        params.cols.classname[0] = "CombinedCanonicalFromFile"
        params.flush()
        #Copy pot_params table
        pot_params = nf.copyNode(sf.root.results.pot_params, comgrp, newname="pot_params")
        tresarr = nf.copyNode(sf.root.results.tresult, comgrp)
        log.debug("K array, first and second order parameters copied.")
        #Only copy foystart if it exists
        if "foystart" in ff.root.results:
            foystarr = nf.copyNode(ff.root.results.foystart, comgrp)
            fotstarr = nf.copyNode(ff.root.results.fotstart, comgrp)
            fotstixarr = nf.copyNode(ff.root.results.fotstartindex, comgrp)
            log.debug("foystart, fotstart, fotstartindex arrays copied.")
        #Copy source terms if it exists, with stepsize 2
        if "sourceterm" in ff.root.results:
            srcarr = nf.copyNode(ff.root.results.sourceterm, comgrp, step=2)
            log.debug("Source term exists in first order file and has been copied.")
        #Get results from first and second order
        fyr = ff.root.results.yresult
        syr = sf.root.results.yresult
        #Begin main loop
        log.debug("Beginning main combination loop...")
        for frow, srow in zip(fyr.iterrows(step=2), syr.iterrows()):
            nrow = np.concatenate((frow, srow))[np.newaxis,...]
            yresarr.append(nrow)
        log.debug("Main combination loop finished.")
        nf.flush()
    finally:
        sf.close()
        ff.close()
        nf.close()
    #Successful execution
    log.debug("First and second order files successfully combined in %s.", newfile)
    return newfile


def main(argv=None):
    """Main function: deal with command line arguments and start calculation as reqd."""
    
    if not argv:
        argv = sys.argv
    
    #Parse command line options
    parser = optparse.OptionParser()
     
    mrggroup = optparse.OptionGroup(parser, "Combination Options",
                        "These options control the combination of first and second order files.")
    mrggroup.add_option("--merge", action="store_true", dest="merge",
                        default=False, help="combine first order and source results in one file")
    mrggroup.add_option("--fofile", action="store", dest="foresults",
                        default=run_config.mrgresults, type="string",
                        metavar="FILE", help="first order or merged fo and src results file, default=%default")
    mrggroup.add_option("--sofile", action="store", dest="soresults",
                        default=run_config.soresults, type="string",
                        metavar="FILE", help="second order results file, default=%default")
    mrggroup.add_option("--cmbfile", action="store", dest="cmbresults",
                        default=run_config.cmbresults, type="string",
                        metavar="FILE", help="new file to store combined results, default=%default")
    parser.add_option_group(mrggroup)
    
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
        
    logfile = os.path.join(run_config.LOGDIR, "cmb.log")
    helpers.startlogging(log, logfile, options.loglevel, consolelevel)
    
    if (not _debug) and (options.loglevel == logging.DEBUG):
        log.warn("Debugging information will not be stored due to setting in run_config.")
        
    if os.path.isfile(options.cmbresults):
        raise IOError("File %s already exists! Please delete or specify another filename." % options.cmbresults)
    
    if not os.path.isdir(os.path.dirname(options.cmbresults)):
        raise IOError("Directory %s does not exist!" % os.path.dirname(options.cmbresults))
    
    try:
        srcfile = combine_results(options.foresults, options.soresults, options.cmbresults)
    except Exception:
        log.exception("Something went wrong while combining files!")
        return 1
        
    return 0
    
#Get root log
if __name__ == "__main__":
    log = logging.getLogger()
    log.name = "cmb"
    log.handlers = []
    sys.exit(main())
else:
    log = logging.getLogger("cmb")
