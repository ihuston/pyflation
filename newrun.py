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

def create_run_directory(newrundir):
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
        
    if bzr_available:
        accelerator_tree, source = BzrDir.open_tree_or_branch(configuration.CODEDIR)
    else:
        pass
    
    return
    
if __name__ == '__main__':
    logging.basicConfig()
    newdir = os.path.join(os.getcwd(), time.strftime("%Y%m%d%H%M%S"))
    print newdir
    try:
        create_run_directory(newdir)
    except Exception, e:
        logging.critical("Something went wrong! Quitting.")
        sys.exit(e)
    
    logging.info("New run directory created successfully.")