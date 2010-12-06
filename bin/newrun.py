''' Script to create and populate new run directory.
Created on 25 Jun 2010

@author: Ian Huston
'''
from pyflation import configuration
import os
import os.path
import helpers
import logging
import sys
import time
from optparse import OptionParser
from setup import setup, setup_args

#Version information
from sys import version as python_version
from numpy import __version__ as numpy_version
from scipy import __version__ as scipy_version
from tables import __version__ as tables_version 


provenance_template = """Provenance document for this Pyflation run
------------------------------------------
                    
Bazaar Revision Control Information (if available)
-------------------------------------------------
Branch name: %(nick)s
Branch revision number: %(revno)s
Branch revision id: %(revid)s
 
Code Directory Information
--------------------------   
Original code directory: %(codedir)s
New run directory: %(newrundir)s
Date run directory was created: %(now)s
       
Version information at time of run creation
-------------------------------------------
Python version: %(python_version)s
Numpy version: %(numpy_version)s
Scipy version: %(scipy_version)s
PyTables version: %(tables_version)s

This information added on: %(now)s.
-----------------------------------------------
        
"""


def create_run_directory(newrundir, codedir, bzr_checkout=False):
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
        import bzrlib.export, bzrlib.workingtree
        bzr_available = True
    except ImportError:
        bzr_available = False
        
    logging.debug("bzr_available=%s", bzr_available)
    
    if bzr_available:
        mytree =  bzrlib.workingtree.WorkingTree.open(codedir)
        if bzr_checkout:
            newtree = mytree.branch.create_checkout(os.path.join(newrundir, 
                            configuration.CODEDIRNAME), lightweight=True)
        else:
            bzrlib.export.export(mytree, os.path.join(newrundir, configuration.CODEDIRNAME))
        
    else:
        raise NotImplementedError("Bazaar is needed to copy code directory. Please do this manually.")
    
    #Try to run setup to create .so files
    try:
        olddir = os.getcwd()
        os.chdir(os.path.join(newrundir, configuration.CODEDIRNAME))
        logging.info("Preparing to compile non-python files.")
        setup(script_args=["build_ext", "-i"], **setup_args)
        os.chdir(olddir)
    except:
        logging.exception("Compiling additional modules did not work. Please do so by hand!")
    
    #Create provenance file detailing revision and branch used
    provenance_dict = dict(python_version=python_version,
                           numpy_version=numpy_version,
                           scipy_version=scipy_version,
                           tables_version=tables_version,
                           codedir=codedir,
                           newrundir=newrundir,
                           now=time.strftime("%Y/%m/%d %H:%M:%S %Z"))
    if bzr_available:
        provenance_dict["nick"] = mytree.branch.nick
        provenance_dict["revno"] = mytree.branch.revno()
        provenance_dict["revid"] = mytree.branch.last_revision()
    else:
        provenance_dict["nick"] = "Unavailable"
        provenance_dict["revno"] = "Unavailable" 
        provenance_dict["revid"] = "Unavailable" 
    provenance_file = os.path.join(newrundir, configuration.LOGDIRNAME, 
                                   configuration.provenancefilename) 
    with open(provenance_file, "w") as f:
        f.write(provenance_template % provenance_dict)
        logging.info("Created provenance file %s." % provenance_file)
    
    return
 

def main(argv = None):
    """Check command line options and start directory creation."""
    if not argv:
        argv = sys.argv
    #Parse command line options
    parser = OptionParser()
    
    parser.set_defaults(loglevel=configuration.LOGLEVEL)
    
    parser.add_option("-d", "--dir", dest="dir", default=os.getcwd(),
                  help="create run directory in DIR, default is current directory", metavar="DIR")
    parser.add_option("-n", "--name", dest="dirname",
                      help="new run directory name")
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
    parser.add_option("--checkout", action="store_true", dest="bzr_checkout",
                      default=True, help="create a bzr checkout instead of export")
        
    (options, args) = parser.parse_args(args=argv[1:])
    
    logging.basicConfig(level=options.loglevel)
        
    if not os.path.isdir(options.dir):
        raise IOError("Please check that parent directory %s exists." % options.dir)

    if options.dirname:
        newdir = os.path.join(options.dir, options.dirname)
        logging.debug("Variable newdir specified with value %s.", newdir)
    else:
        newdir = os.path.join(options.dir, time.strftime("%Y%m%d%H%M%S"))
        logging.debug("Variable newdir created with value %s", newdir)
    
    if options.codedir:
        codedir = options.codedir
        logging.debug("Option codedir specified with value %s.", options.codedir)
    else:
        codedir = os.path.join(configuration.CODEDIR)
        logging.debug("Variable codedir created with value %s.", codedir)
        
    try:
        create_run_directory(newdir, codedir, options.bzr_checkout)
    except Exception, e:
        logging.critical("Something went wrong! Quitting.")
        sys.exit(e)
    
    logging.info("New run directory created successfully.")
    return

if __name__ == '__main__':
    main()
    