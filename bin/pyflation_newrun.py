#!/usr/bin/env python
"""pyflation_newrun - Script to create and populate new Pyflation run directory.


"""
#Author: Ian Huston
#For license and copyright information see LICENSE.txt which was distributed with this file.


import os.path
import logging
import sys
import time
from optparse import OptionParser
import shutil

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext #@UnresolvedImport
import numpy

#Version information
from sys import version as python_version
from numpy import __version__ as numpy_version
from scipy import __version__ as scipy_version
from tables import __version__ as tables_version 
from Cython.Compiler.Version import version as cython_version




try:
    #Local modules from pyflation package
    from pyflation import __version__ as pyflation_version
    from pyflation import configuration, helpers
except ImportError,e:
    if __name__ == "__main__":
        msg = """Pyflation module needs to be available. 
Either run this script from the base directory as bin/newrun.py or add directory enclosing pyflation package to PYTHONPATH."""
        print msg, e
        sys.exit(1)
    else:
        raise
    


# Test whether Bazaar is available
try:
    import bzrlib.export, bzrlib.workingtree
    bzr_available = True
except ImportError:
    bzr_available = False

provenance_template = """Provenance document for this Pyflation run
------------------------------------------

Pyflation Version
-----------------
Version: %(version)s
                    
%(bzrinfo)s
 
Code Directory Information
--------------------------   
Original code directory: %(codedir)s
New run directory: %(newrundir)s
Date run directory was created: %(now)s
       
Library version information at time of run creation
-------------------------------------------
Python version: %(python_version)s
Numpy version: %(numpy_version)s
Scipy version: %(scipy_version)s
PyTables version: %(tables_version)s
Cython version: %(cython_version)s

This information added on: %(now)s.
-----------------------------------------------
        
"""

def copy_code_directory(codedir, newcodedir, use_bzr=False, bzr_available=False):
    """Create code directory and copy from Bazaar repository."""
    mytree = None
    if bzr_available and use_bzr:
        try:            
            mytree =  bzrlib.workingtree.WorkingTree.open(codedir)
            newtree = mytree.branch.create_checkout(newcodedir, lightweight=True)
            logging.debug("Bazaar code copied successfully.")
        except bzrlib.errors.NotBranchError, e:
            mytree = None
            logging.error("Error using bazaar. Directory %s is not a branch." % codedir)
    else:
        if use_bzr:
            logging.error("Bazaar not available, copying code instead.")
        try:
            shutil.copytree(codedir, newcodedir)
            logging.debug("Copying of code successful.")
        except:
            logging.error("Error copying code. Please do so manually.")
    
    
    ext_modules = [Extension("pyflation.sourceterm.srccython", ["pyflation/sourceterm/srccython.pyx"], 
               extra_compile_args=["-g"], 
               extra_link_args=["-g"],
               include_dirs=[numpy.get_include()]),
               #
               Extension("pyflation.romberg", ["pyflation/romberg.pyx"], 
               extra_compile_args=["-g"], 
               extra_link_args=["-g"],
               include_dirs=[numpy.get_include()]),
               #
               ]
    
    setup_args = dict(name='Pyflation',
                  author='Ian Huston',
                  author_email='ian.huston@gmail.com',
                  url='http://www.maths.qmul.ac.uk/~ith/pyflation',
                  description='Cosmological Inflation in Python',
                  cmdclass = {'build_ext': build_ext},
                  ext_modules = ext_modules)
    
    #Try to run setup to create .so files
    try:
        olddir = os.getcwd()
        os.chdir(newcodedir)
        logging.info("Preparing to compile non-python files.")
        setup(script_args=["build_ext", "-i"], **setup_args)
        os.chdir(olddir)
    except:
        logging.exception("Compiling additional modules did not work. Please do so by hand!")
    
    return mytree

def create_run_directory(newrundir, codedir, copy_code=False, 
                         use_bzr=False):
    """Create the run directory using `newdir` as directory name."""
    if os.path.isdir(newrundir):
        raise IOError("New run directory already exists!")
    
    try:
        helpers.ensurepath(newrundir)
    except OSError:
        logging.error("Cannot create parent directories for run.")
        raise
    
    mytree = None
    if copy_code:
        #Check for bzr
        logging.debug("bzr_available=%s", bzr_available)
        newcodedir = os.path.join(newrundir, configuration.CODEDIRNAME)
        logging.debug("Attempting to copy code directory...")
        mytree = copy_code_directory(codedir, newcodedir, use_bzr, bzr_available)    
    else:
        logging.debug("No copying of code directory attempted.")
    
    try:
        if not os.path.isdir(newrundir):
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
        
    #Copy run_config template file into new run directory if not already
    #copied by bzr.
    if not os.path.isfile(os.path.join(newrundir, "run_config.py")):
        logging.debug("Need to create run_config.py from template.")
        pkgdir = os.path.dirname(os.path.abspath(configuration.__file__))
        runconfigtemplate = os.path.join(pkgdir, configuration.RUNCONFIGTEMPLATE)
        logging.debug("run_config template file is %s" % runconfigtemplate)
        if not os.path.isfile(runconfigtemplate):
            raise IOError("File run_config.template is not available to be copied!") 
        else:
            try:
                shutil.copyfile(runconfigtemplate, os.path.join(newrundir, "run_config.py"))
                logging.debug("run_config file copied successfully.")
            except:
                logging.error("Error copying run_config template file.")
                raise
    else:
        logging.debug("The run_config.py file already exists in correct directory.")
    
    #Create provenance file detailing revision and branch used
    prov_dict = dict(version=pyflation_version,
                     python_version=python_version,
                     numpy_version=numpy_version,
                     scipy_version=scipy_version,
                     tables_version=tables_version,
                     cython_version=cython_version,
                     codedir=codedir,
                     newrundir=newrundir,
                     now=time.strftime("%Y/%m/%d %H:%M:%S %Z"))
    if mytree:
        prov_dict["bzrinfo"] = """
Bazaar Revision Control Information
-------------------------------------------------
Branch name: %(nick)s
Branch revision number: %(revno)s
Branch revision id: %(revid)s""" % {"nick": mytree.branch.nick,
                                    "revno": mytree.branch.revno(),
                                    "revid": mytree.branch.last_revision()}
    else:
        prov_dict["bzrinfo"] = ""
         
    provenance_file = os.path.join(newrundir, configuration.LOGDIRNAME, 
                                   configuration.provenancefilename) 
    with open(provenance_file, "w") as f:
        f.write(provenance_template % prov_dict)
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
    parser.add_option("-n", "--name", dest="dirname", default="pyflation_run",
                      help="new run directory name, default is pyflation_run")
    parser.add_option("-c", "--codedir", dest="codedir",
                  help="copy code from CODEDIR (where run_config.py resides)", metavar="CODEDIR")
    parser.add_option("-q", "--quiet",
                  action="store_const", const=logging.FATAL, dest="loglevel", 
                  help="only print fatal error messages")
    parser.add_option("-v", "--verbose",
                  action="store_const", const=logging.INFO, dest="loglevel", 
                  help="print informative messages")
    parser.add_option("--debug",
                  action="store_const", const=logging.DEBUG, dest="loglevel", 
                  help="print lots of debugging information")
    parser.add_option("--copy-code", action="store_true", dest="copy_code",
                      default=False, help="copy code directory into run directory (using Bazaar)")
    parser.add_option("--bzr", action="store_true", dest="use_bzr",
                      default=False, help="use Bazaar to create branch of code")
        
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
    
    if options.codedir and options.copy_code:
        codedir = os.path.abspath(options.codedir)
        logging.debug("Option codedir specified with value %s.", options.codedir)
    elif options.copy_code:
        try:
            from run_config import CODEDIR
        except ImportError, e:
            msg = "Configuration file run_config.py needs to be available. \
                Use --codedir command line option to specify its location."
            print msg, e
            sys.exit(1)
        codedir = os.path.abspath(CODEDIR)
        logging.debug("Variable codedir created with value %s.", codedir)
    else:
        codedir = None
        
    try:
        create_run_directory(newdir, codedir, options.copy_code,
                             options.use_bzr)
    except Exception, e:
        logging.critical("Something went wrong! Quitting.")
        sys.exit(e)
    
    logging.info("New run directory created successfully.")
    return

if __name__ == '__main__':
    main()
    