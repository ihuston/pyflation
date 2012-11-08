"""
Helper module to provide the make_wrapper_model function.

The :func:`make_wrapper_model` function takes a filename and returns a model instance
corresponding to the one stored in the file. This allows easier access to the
results than through a direct inspection of the HDF5 results file.


"""
#Author: Ian Huston
#For license and copyright information see LICENSE.txt which was distributed with this file.



from __future__ import division

#system modules
import os.path
import tables 
import logging

#local modules from pyflation
from configuration import _debug
from pyflation import cmpotentials
from pyflation import cosmomodels as c
from pyflation import reheating

#Start logging
root_log_name = logging.getLogger().name
module_logger = logging.getLogger(root_log_name + "." + __name__)

modules_with_classes = [c, reheating]


def make_wrapper_model(modelfile, *args, **kwargs):
    """Return a wrapper class that provides the given model class from a file."""
    #Check file exists
    if not os.path.isfile(modelfile):
        raise IOError("File does not exist!")
    try:
        rf = tables.openFile(modelfile, "r")
        try:
            try:
                params = rf.root.results.parameters
                modelclassname = params[0]["classname"]
            except tables.NoSuchNodeError:
                raise c.ModelError("File does not contain correct model data structure!")
        finally:
            rf.close()
    except IOError:
        raise
    try:
        for module in modules_with_classes:
            modelclass = getattr(module, modelclassname, None)
            if modelclass:
                break #We have found class in one of the modules, stop looking
    except AttributeError:
        raise c.ModelError("Model class does not exist!")
                
    class ModelWrapper(modelclass):
        """Wraps first order model using HDF5 file of results."""
        
        def __init__(self, filename, *args, **kwargs):
            """Get results from file and instantiate variables.
            Opens file with handle saved as self._rf. File is closed in __del__"""
            #Call super class __init__ method
            super(ModelWrapper, self).__init__(*args, **kwargs)
            
            #Check file exists
            if not os.path.isfile(filename):
                raise IOError("File does not exist!")
            try:
                if _debug:
                    self._log.debug("Opening file " + filename + " to read results.")
                try:
                    self._rf = tables.openFile(filename, "r")
                    results = self._rf.root.results
                    self.yresult = results.yresult
                    self.tresult = results.tresult
                    if "bgystart" in results:
                        self.bgystart = results.bgystart
                    if "ystart" in results:
                        self.ystart = results.ystart
                    self.fotstart = results.fotstart
                    if "fotstartindex" in results:
                        #for backwards compatability only set if it exists
                        self.fotstartindex = results.fotstartindex
                    self.foystart = results.foystart
                    self.k = results.k[:]
                    params = results.parameters
                except tables.NoSuchNodeError:
                    raise c.ModelError("File does not contain correct model data structure!")
                try:
                    self.source = results.sourceterm
                except tables.NoSuchNodeError:
                    if _debug:
                        self._log.debug("First order file does not have a source term.")
                    self.source = None
                # Put potential parameters into right variable
                try:
                    potparamstab = results.pot_params
                    for row in potparamstab:
                        key = row["name"]
                        val = row["value"]
                        self.pot_params[key] = val
                except tables.NoSuchNodeError:
                    if _debug:
                        self._log.debug("No pot_params table in file.") 
                        
                # Put provenance values into right variable
                if not hasattr(self, "provenance"):
                    self.provenance = {}
                try:
                    provtab = results.provenance
                    for row in provtab:
                        key = row["name"]
                        val = row["value"]
                        self.provenance[key] = val
                except tables.NoSuchNodeError:
                    if _debug:
                        self._log.debug("No provenance table in file.")               
                
                #Put params in right slots
                for ix, val in enumerate(params[0]):
                    self.__setattr__(params.colnames[ix], val)
                #set correct potential function (only works with cmpotentials currently)
                self.potentials = getattr(cmpotentials, self.potential_func)
            except IOError:
                raise
            
            #Set indices correctly
            self.H_ix = self.nfields*2
            self.bg_ix = slice(0,self.nfields*2+1)
            self.phis_ix = slice(0,self.nfields*2,2)
            self.phidots_ix = slice(1,self.nfields*2,2)
            self.pert_ix = slice(self.nfields*2+1, None)
            self.dps_ix = slice(self.nfields*2+1, None, 2)
            self.dpdots_ix = slice(self.nfields*2+2, None, 2)
            
            #Fix bgmodel to actual instance
            if self.ystart is not None:
                #Check ystart is in right form (1-d array of three values)
                if len(self.ystart.shape) == 1:
                    ys = self.ystart[self.bg_ix]
                elif len(self.ystart.shape) == 2:
                    ys = self.ystart[self.bg_ix,0]
            else:
                ys = results.bgresults.yresult[0]
            self.bgmodel = self.bgclass(ystart=ys, tstart=self.tstart, tend=self.tend, 
                            tstep_wanted=self.tstep_wanted, solver=self.solver,
                            potential_func=self.potential_func, 
                            nfields=self.nfields, pot_params=self.pot_params)
            #Put in data
            try:
                if _debug:
                    self._log.debug("Trying to get background results...")
                self.bgmodel.tresult = self._rf.root.bgresults.tresult[:]
                self.bgmodel.yresult = self._rf.root.bgresults.yresult
            except tables.NoSuchNodeError:
                raise c.ModelError("File does not contain background results!")
            #Get epsilon
            if _debug:
                self._log.debug("Calculating self.bgepsilon...")
            self.bgepsilon = self.bgmodel.getepsilon()
            #Success
            self._log.info("Successfully imported data from file into model instance.")
        
        def __del__(self):
            """Close file when object destroyed."""
            try:
                if _debug:
                    self._log.debug("Trying to close file...")
                self._rf.close()
            except IOError:
                raise
    return ModelWrapper(modelfile, *args, **kwargs)