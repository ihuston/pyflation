"""Second order helper functions by Ian Huston
    $Id: sohelpers.py,v 1.8 2009/03/17 18:42:26 ith Exp $
    
    Provides helper functions for second order data from cosmomodels.py"""
    
import tables
import numpy as N
import cosmomodels as c
import logging
import time
import os.path

_log = logging.getLogger(__name__)

def combine_source_and_fofile(sourcefile, fofile, newfile=None):
    """Copy source term to first order file in preparation for second order run.
    
    Parameters
    ----------
    sourcefile: String
                Full path and filename of source file.
                
    fofile: String
            Full path and filename of first order results file.
            
    newfile: String, optional
             Full path and filename of combined file to be created.
             Default is to place the new file in the same directory as the source
             file with the filename having "src" replaced by "foandsrc".
             
    Returns
    -------
    newfile: String
             Full path and filename of saved combined results file.
    """
    if not newfile or not os.path.isdir(os.path.dirname(newfile)):
        newfile = os.path.dirname(fofile) + os.sep + os.path.basename(sourcefile).replace("src", "foandsrc")
        _log.info("Saving combined source and first order results in file %s.", newfile)
    try:
        sf = tables.openFile(sourcefile, "r")
        ff = tables.openFile(fofile, "r")
        nf = tables.openFile(newfile, "a")
    except IOError:
        _log.exception("Source or first order files not found!")
        raise
    
    try:
        try:
            sterm = sf.root.results.sourceterm
            srck = sf.root.results.k
            srcnix = sf.root.results.nix
        except tables.NoSuchNodeError:
            _log.exception("Source term file not in correct format!")
            raise
        fres = ff.root.results
        nres = nf.copyNode(ff.root.results, nf.root)
        #Check that all time steps are calculated
        if len(fres.tresult) != len(srcnix):
            raise ValueError("Not all timesteps have had source term calculated!")
        #Copy first order results
        numks = len(srck)
        yres = fres.yresult.copy(nres, stop=numks)
        tres = fres.tresult.copy(nres)
        tstart = fres.fotstart.copy(nres, stop=numks)
        ystart = fres.foystart.copy(nres, stop=numks)
        params = fres.parameters.copy(nres)
        bgres = nf.copyNode(ff.root.bgresults, nf.root, recursive=True)
        #Copy source term
        nf.copyNode(sterm, nres)
        #Copy source k range
        nf.copyNode(srck, nres)
        _log.info("Source term successfully copied to new file %s.", newfile)
    finally:
        sf.close()
        ff.close()
        nf.close()
    return newfile
        
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
    