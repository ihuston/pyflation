"""srcmerge.py - Combine sourcefiles in a directory into one master file by n index
Author: Ian Huston
"""
import tables
import numpy as np
import os
import logging
import re

import run_config

from run_config import _debug
import sys
import helpers
import optparse


def mergefiles(newfile, dirname, pattern=None):
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
    
    if not os.path.isdir(dirname):
        raise IOError("Directory %s does not exist!" % dirname)
    
    if os.path.isfile(newfile):
        raise IOError("File %s already exists! Please delete or specify another file." % newfile)
    if not os.path.isdir(os.path.dirname(newfile)):
        raise IOError("Directory %s does not exist for file to be saved in." % os.path.dirname(newfile))
    
    #Check and compute regex pattern
    if not pattern:
        pattern = run_config.pattern
    if _debug:
        log.debug("Regex pattern is %s" % pattern)
    regex = re.compile(pattern)
    
    filenames = []
    for f in os.listdir(dirname):
        if os.path.isfile(os.path.join(dirname, f)):
            if regex.search(f):
                filenames.append(os.path.join(dirname, f))
        
    try:
        #Open all hf5 files in directory
        files = [tables.openFile("".join([dirname, os.path.sep, f])) for f in filenames]
    except IOError:
        log.error("Cannot open hf5 files in %s!", dirname)
        raise

        
    try:
        nf = tables.openFile(newfile, "a")
    except IOError:
        log.error("Error opening combined source file %s!", newfile)
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
                raise ValueError("Source files do not cover entire range of time steps! ns=%d nendprev=%d" % (ns, nendprev))
            nendprev = ne
        #Copy first sourceterm, k and nix arrays
        nres = nf.copyNode(indexlist[0][0], nf.root, recursive=True)
        #Copy rest of sourceterm and nix arrays from list
        log.info("Copying sourceterm and tstep arrays...")
        for fres, ns, ne in indexlist[1:]:
            nres.nix.append(fres.nix[:])
            nres.sourceterm.append(fres.sourceterm[:])
        log.info("Merging successful!")    
    finally:
        res = [f.close() for f in files]
        nf.close()
    return newfile


def main(argv=None):
    """Main function: deal with command line arguments and start calculation as reqd."""
    
    if not argv:
        argv = sys.argv
    
    #Parse command line options
    parser = optparse.OptionParser()
    
    parser.add_option("-f", "--filename", action="store", dest="mrgresults", 
                      default=run_config.mrgresults, type="string", 
                      metavar="FILE", help="file to store merged results, default=%default")
    parser.add_option("-d", "--dirname", action="store", dest="dirname",
                      default=run_config.RESULTSDIR, type="string",
                      metavar="DIR", help="directory to search for src node files, " 
                      "default=%default")  
    parser.add_option("--pattern", action="store", dest="pattern",
                      default=run_config.pattern, type="string",
                      help="regex pattern to match with src node files, default %default")
    
    loggroup = optparse.OptionGroup(parser, "Log Options", 
                           "These options affect the verbosity of the log files generated.")
    loggroup.add_option("-q", "--quiet",
                  action="store_const", const=logging.FATAL, dest="loglevel", 
                  help="only print fatal error messages")
    loggroup.add_option("-v", "--verbose",
                  action="store_const", const=logging.INFO, dest="loglevel", 
                  help="print informative messages")
    loggroup.add_option("--debug",
                  action="store_const", const=logging.DEBUG, dest="loglevel", 
                  help="log lots of debugging information",
                  default=run_config.LOGLEVEL)
    loggroup.add_option("--console", action="store_true", dest="console",
                        default=False, help="if selected matches console log level " 
                        "to selected file log level, otherwise only warnings are shown.")
    parser.add_option_group(loggroup)
    
    (options, args) = parser.parse_args(args=argv[1:])
        
            
    #Start the logging module
    if options.console:
        consolelevel = options.loglevel
    else:
        consolelevel = logging.WARN
        
    logfile = os.path.join(run_config.LOGDIR, "src.log")
    helpers.startlogging(log, logfile, options.loglevel, consolelevel)
    
    if (not _debug) and (options.loglevel == logging.DEBUG):
        log.warn("Debugging information will not be stored due to setting in run_config.")
        
    if os.path.isfile(options.mrgresults):
        raise IOError("File %s already exists! Please delete or specify another filename." % options.mrgresults)
    
    if not os.path.isdir(options.dirname):
        raise IOError("Directory %s does not exist!" % options.dirname)
    
    try:
        mergefiles(options.mrgresults, options.dirname, options.pattern)
    except Exception:
        log.exception("Something went wrong while merging results!")
        return 1
    
    return 0
    
#Get root log
if __name__ == "__main__":
    log = logging.getLogger("srcmerge")
    log.handlers = []
    sys.exit(main())
else:
    log = logging.getLogger(__name__)
    
