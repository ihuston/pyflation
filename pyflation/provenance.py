"""pyflation.provenance - Information about the provenance of the data and code
created by pyflation.

"""
#Author: Ian Huston
#For license and copyright information see LICENSE.txt which was distributed with this file.


import os.path
import logging
import sys
import time

#Version information
from sys import version as python_version
from numpy import __version__ as numpy_version
from scipy import __version__ as scipy_version
from tables import __version__ as tables_version 
from Cython.Compiler.Version import version as cython_version




#Local modules from pyflation package
from pyflation import __version__ as pyflation_version
from pyflation import configuration
    


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



def provenance(newrundir, codedir, mytree):
    prov_dict = dict(version=pyflation_version, python_version=python_version, 
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
Branch revision id: %(revid)s""" % {"nick":mytree.branch.nick, 
            "revno":mytree.branch.revno(), 
            "revid":mytree.branch.last_revision()}
    else:
        prov_dict["bzrinfo"] = ""
    return newrundir, prov_dict

def write_provenance_file(newrundir, codedir, mytree):
    #Create provenance file detailing revision and branch used
    newrundir, prov_dict = provenance(newrundir, codedir, mytree)
         
    provenance_file = os.path.join(newrundir, configuration.LOGDIRNAME, 
                                   configuration.provenancefilename) 
    with open(provenance_file, "w") as f:
        f.write(provenance_template % prov_dict)
        logging.info("Created provenance file %s." % provenance_file)
        
    return provenance_file