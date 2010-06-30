''' Script to create and populate new run directory.
Created on 25 Jun 2010

@author: Ian Huston
'''
import configuration
import os
import os.path
import helpers
import logging
import sys
import time
from optparse import OptionParser

def create_run_directory(newrundir, codedir):
    """Create the run directory using `newdir` as directory name."""
    if os.path.isdir(newrundir):
        raise IOError("New run directory already exists!")
    
    try:
        helpers.ensurepath(newrundir)
        os.makedirs(newrundir)
    except OSError:
        logging.error("Creating new run directory failed.")
        raise
    
    resultsdir = os.path.join(newrundir, configuration.RESULTSDIRNAME)
    logdir = os.path.join(newrundir, configuration.LOGDIRNAME)
    qsublogsdir = os.path.join(newrundir, configuration.QSUBLOGSDIRNAME)
    qsubscriptsdir = os.path.join(newrundir, configuration.QSUBSCRIPTSDIRNAME)
    #debug info
    logging.debug("resultsdir=%s, logdir=%s, qsublogsdir=%s, qsubscriptsdir=%s",
                  resultsdir, logdir, qsublogsdir, qsubscriptsdir)
    
    try:
        os.makedirs(resultsdir)
        os.makedirs(logdir)
        os.makedirs(qsublogsdir)
        os.makedirs(qsubscriptsdir)
    except OSError:
        logging.error("Creating subdirectories in new run directory failed.")
        raise
    
    #Try to do Bazaar checkout of code
    try:
        from bzrlib.bzrdir import BzrDir
        bzr_available = True
    except ImportError:
        bzr_available = False
        
    logging.debug("bzr_available=%s", bzr_available)
    
    if bzr_available:
        accelerator_tree, source = BzrDir.open_tree_or_branch(codedir)
        source.create_checkout(os.path.join(newrundir, configuration.CODEDIRNAME), 
                               None, True, accelerator_tree)
        
    else:
        raise NotImplementedError("Bazaar is needed to copy code directory. Please do this manually.")
    
    return
    
def main():
    """Check command line options and start directory creation."""
    
    #Parse command line options
    parser = OptionParser()
    parser.add_option("-n", "--newdir", dest="newdir",
                  help="create run directory in NEWDIR", metavar="NEWDIR")
    parser.add_option("-c", "--codedir", dest="codedir",
                  help="copy code from CODEDIR", metavar="CODEDIR")
    parser.add_option("-q", "--quiet",
                  action="store_const", const=logging.FATAL, dest="loglevel", 
                  help="only print fatal error messages")
    parser.add_option("-v", "--verbose",
                  action="store_const", const=logging.INFO, dest="loglevel", 
                  help="print informative messages")
    parser.add_option("--debug",
                  action="store_const", const=logging.DEBUG, dest="loglevel", 
                  help="print lots of debugging information")
        
    (options, args) = parser.parse_args()
    
    logging.basicConfig(level=options.loglevel)
        
    if options.newdir:
        newdir = options.newdir
        logging.debug("Option newdir specified with value %s.", options.newdir)
    else:
        newdir = os.path.join(configuration.BASEDIR, 
                              configuration.RUNDIRNAME, time.strftime("%Y%m%d%H%M%S"))
        logging.debug("Variable newdir created with value %s", newdir)
    
    if options.codedir:
        codedir = options.codedir
        logging.debug("Option codedir specified with value %s.", options.codedir)
    else:
        codedir = os.path.join(configuration.BZRCODEDIR)
        logging.debug("Variable codedir created with value %s.", codedir)
        
    try:
        create_run_directory(newdir, codedir)
    except Exception, e:
        logging.critical("Something went wrong! Quitting.")
        sys.exit(e)
    
    logging.info("New run directory created successfully.")
    return

if __name__ == '__main__':
    main()
    