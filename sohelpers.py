"""Second order helper functions by Ian Huston
    $Id: sohelpers.py,v 1.5 2009/01/29 19:16:41 ith Exp $
    
    Provides helper functions for second order data from cosmomodels.py"""
    
import tables
import numpy as N
import cosmomodels as c
import logging
import time
import os.path

_log = logging.getLogger(__name__)

def copy_source_to_fofile(sourcefile, fofile):
    """Copy source term to first order file in preparation for second order run."""
    try:
        sf = tables.openFile(sourcefile, "r")
        ff = tables.openFile(fofile, "a")
    except IOError:
        _log.exception("Source or first order files not found!")
        raise
    
    try:
        try:
            sterm = sf.root.results.sourceterm
        except NoSuchNodeError:
            _log.exception("Source term not found in file!")
            raise
        fres = ff.root.results
        ff.copyNode(sterm, fres)
        _log.info("Source term successfully copied to first order file %s.", fofile)
    finally:
        sf.close()
        ff.close()
        
def combine_results(fofile, sofile, newfile=None):
    """Combine the first and second order results from given files, and save in newfile."""
    if not newfile:
        now = time.strftime("%Y%m%d%H%M")
        newfile = c.RESULTS_PATH + "cmb" + now + ".hf5"
        _log.info("Filename set to " + newfile)
        
    if os.path.isdir(os.path.dirname(newfile)):
        if os.path.isfile(newfile):
            raise IOError("File already exists!")
        else:
            _log.debug("File does not exist, using write mode.")
            filemode = "w" #Writing to new file
    else:
        raise IOError("Directory 'results' does not exist")
    #Add compression
    filters = tables.Filters(complevel=1, complib="zlib")
    try:
        sf = tables.openFile(sofile, "r")
        ff = tables.openFile(fofile, "r")
        nf = tables.openFile(newfile, filemode, filters=filters)
    except IOError:
        _log.exception("Error opening files!")
        raise
    try:
        #Create groups required
        comgrp = nf.createGroup(nf.root, "results", "Combined first and second order results")
        _log.debug("Results group created in combined file.")
        #Store bg results:
        bggrp = nf.copyNode(ff.root.bgresults, nf.root, recursive=True)
        _log.debug("Bg results copied.")
        #Save results
        oldyshape = list(sf.root.results.yresult[:,:,0:0].shape) #2nd order shape
        oldyshape[1] += ff.root.results.yresult.shape[1] #add number of 1st order vars
        yresarr = nf.createEArray(comgrp, "yresult", tables.Float64Atom(), oldyshape, filters=filters, chunkshape=(10,7,10))
        _log.debug("New yresult array with shape %s created.", str(oldyshape))
        #Copy other important arrays
        karr = nf.copyNode(sf.root.results.k, comgrp)
        foparams = nf.copyNode(ff.root.results.parameters, comgrp, newname="foparameters")
        soparams = nf.copyNode(sf.root.results.parameters, comgrp, newname="soparameters")
        #Copy parameters and change classname for compatibility
        params = nf.copyNode(sf.root.results.parameters, comgrp, newname="parameters")
        params.cols.classname[0] = "CombinedCanonicalFromFile"
        params.flush()
        tresarr = nf.copyNode(sf.root.results.tresult, comgrp)
        _log.debug("K array, first and second order parameters copied.")
        #Only copy foystart if it exists
        if "foystart" in ff.root.results:
            foystarr = nf.copyNode(ff.root.results.foystart, comgrp)
            fotstarr = nf.copyNode(ff.root.results.fotstart, comgrp)
            _log.debug("foystart, fotstart arrays copied.")
        #Get results from first and second order
        fyr = ff.root.results.yresult
        syr = sf.root.results.yresult
        #Begin main loop
        _log.debug("Beginning main combination loop...")
        for frow, srow in zip(fyr.iterrows(), syr.iterrows()):
            nrow = N.concatenate((frow[:,::2], srow)).transpose()[..., N.newaxis]
            yresarr.append(nrow)
        _log.debug("Main combination loop finished.")
        nf.flush()
    finally:
        sf.close()
        ff.close()
        nf.close()
    #Successful execution
    _log.debug("First and second order files successfully combined in %s.", newfile)
    return newfile
    
    
    
       
        
        
                    
    