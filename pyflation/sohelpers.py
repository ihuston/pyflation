"""sohelpers.py - Second order helper functions

Provides helper functions for second order data from cosmomodels.py.

"""
#Author: Ian Huston
#For license and copyright information see LICENSE.txt which was distributed with this file.


    
import tables
import logging
import os.path

#Start logging
root_log_name = logging.getLogger().name
_log = logging.getLogger(root_log_name + "." + __name__)


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
    if os.path.isfile(newfile):
        newfile = newfile.replace("foandsrc", "foandsrc_")
    try:
        sf = tables.openFile(sourcefile, "r")
        ff = tables.openFile(fofile, "r")
        nf = tables.openFile(newfile, "w") #Write to new file, not append to old.
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
        #List of things to copy
        numkscopylist = [fres.yresult, fres.fotstart, fres.fotstartindex,
                         fres.foystart]
        for arr in numkscopylist:
            nf.createArray(nres, arr.name, arr[...,:numks])
        tres = fres.tresult.copy(nres)
        params = fres.parameters.copy(nres)
        pot_params = fres.pot_params.copy(nres)
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
