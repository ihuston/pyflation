"""pyflation.provenance - Information about the provenance of the data and code
created by pyflation.

"""
#Author: Ian Huston
#For license and copyright information see LICENSE.txt which was distributed with this file.


import os.path
import logging
import sys
import time
import subprocess

#Version information
from sys import version as python_version
from numpy import __version__ as numpy_version
from scipy import __version__ as scipy_version
from tables import __version__ as tables_version 
from Cython.Compiler.Version import version as cython_version

#Local modules from pyflation package
from pyflation import __version__ as pyflation_version
from pyflation import configuration

run_creation_template = """Provenance document for this Pyflation run
------------------------------------------

Pyflation Version
-----------------
Version: %(pyflation_version)s
Git revision: %(git_revision)s
                    
 
Code Directory Information
--------------------------   
Original code directory: %(codedir)s
New run directory: %(newrundir)s
Date run directory was created: %(timestamp)s
       
Library version information at time of run creation
-------------------------------------------
Python version: %(python_version)s
Numpy version: %(numpy_version)s
Scipy version: %(scipy_version)s
PyTables version: %(tables_version)s
Cython version: %(cython_version)s

This information added on: %(timestamp)s.
-----------------------------------------------
        
"""



def provenance(newrundir, codedir):
    
    #Get git revision number if available
    proc= subprocess.Popen("git rev-parse HEAD", stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE, shell=True)
    response = proc.communicate()
    if response[0] == '' or response[1] != '':
        git_revision = "Not available"
    else:
        git_revision = response[0].strip()
    
    prov_dict = dict(pyflation_version=pyflation_version, 
                     python_version=python_version, 
                     numpy_version=numpy_version, 
                     scipy_version=scipy_version, 
                     tables_version=tables_version, 
                     cython_version=cython_version,
                     git_revision=git_revision, 
                     codedir=codedir, 
                     newrundir=newrundir, 
                     timestamp=time.strftime("%Y/%m/%d %H:%M:%S %Z"))
    
    return prov_dict

def write_provenance_file(newrundir, codedir):
    #Create provenance file detailing revision and branch used
    prov_dict = provenance(newrundir, codedir)
         
    provenance_file = os.path.join(newrundir, configuration.LOGDIRNAME, 
                                   configuration.provenancefilename) 
    with open(provenance_file, "w") as f:
        f.write(run_creation_template % prov_dict)
        logging.info("Created provenance file %s." % provenance_file)
        
    return provenance_file