"""srcmerge.py - Combine sourcefiles in a directory into one master file by n index
Author: Ian Huston
"""
import tables
import numpy as np
import os
import logging
import logging.handlers
from hconfig import *

def startlogging():
    """Start the logging system to store rotational log based on date."""
    logger.setLevel(LOGLEVEL)
    #Get date for logfile
    date = time.strftime("%Y%m%d")
    #create file handler and set level to debug
    fh = logging.handlers.RotatingFileHandler(filename=LOGDIR + date + ".log", maxBytes=2**20, backupCount=50)
    fh.setLevel(LOGLEVEL)
    #create console handler and set level to error
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    #create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    #add formatter to fh
    fh.setFormatter(formatter)
    #add formatter to ch
    ch.setFormatter(formatter)
    #add fh to logger
    logger.addHandler(fh)
    #add ch to logger
    logger.addHandler(ch)
    logger.debug("Logging started.")
    
def mergefiles(newfile=None, dirname=None):
    """Merge sourceterms in `dirname` into one file `newfile`
    
    Parameters
    ----------
    newfile: string, optional
             Full filename of source file to be created.
             
    dirname: string, optional
             Directory to look for source files in. Defaults to current directory.
    
    Results
    -------
    newfile: string
             Filename of combined source file.
    """
    if not dirname:
        dirname = os.getcwd()
    if not newfile or not os.path.isdir(os.path.dirname(newfile)) or os.path.isfile(newfile):
        newfile = "".join([dirname, os.path.sep, "src-combined.hf5"]) 
    filenames = [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f)) and os.path.splitext(f)[1] == ".hf5" and "src" in f[:3]]
    try:
        #Open all hf5 files in directory
        files = [tables.openFile("".join([dirname, os.path.sep, f])) for f in filenames]
    except IOError:
        logger.error("Cannot open hf5 files in %s!", dirname)
        raise
    try:
        nf = tables.openFile(newfile, "a")
    except IOError:
        logger.error("Error opening combined source file %s!", newfile)
    try:
        #Check that all k ranges are the same
        ks = [f.root.results.k[:] for f in files]
        if not np.all([np.all(k == ks[0]) for k in ks]):
            raise ValueError("Not all source files use same k range!")
        #Construct list of sourceterms, start and end indices
        indexlist = [(f.root.results, f.root.results.nix[0], f.root.results.nix[-1]) for f in files]
        #Comparison function for nix[0]
        fcmp = lambda a,b: cmp(int(a[1]),int(b[1]))
        indexlist.sort(cmp=fcmp) #Sort list in place
        #Check all the tsteps are present
        nendprev = indexlist[0][1] - 1
        for fres, ns, ne in indexlist:
            if ns != nendprev + 1:
                raise ValueError("Source files do not cover entire range of time steps!")
            nendprev = ne
        #Copy first sourceterm, k and nix arrays
        nres = nf.copyNode(indexlist[0][0], nf.root, recursive=True)
        #Copy rest of sourceterm and nix arrays from list
        logger.info("Copying sourceterm and tstep arrays...")
        for fres, ns, ne in indexlist[1:]:
            nres.nix.append(fres.nix[:])
            nres.sourceterm.append(fres.sourceterm[:])
        logger.info("Merging successful!")    
    finally:
        res = [f.close() for f in files]
        nf.close()
    return newfile
        
#Get root logger
if __name__ == "__main__":
    logger = logging.getLogger()
    logger.handlers = []
    startlogging()
    mergefiles()
else:
    logger = logging.getLogger(__name__)
    
