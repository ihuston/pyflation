# -*- coding: utf-8 -*-
"""start.py - Generate bash scripts for qsub and execute.
Author: Ian Huston
"""
import os.path
import configuration
import harness
import time
import sys
from helpers import ensurepath
from optparse import OptionParser
import logging

fotemplatefile = os.path.join(configuration.CODEDIR, "forun-template.sh")
fulltemplatefile = os.path.join(configuration.CODEDIR, "full-template.sh")

        
def genfullscripts(tfilename):
    numsoks = configuration.NUMSOKS
    try:
        f = open(tfilename, "r")
    except IOError:
        raise
    try:
        text = f.read()
        for d in sublist:
            #Put k values in dictionary
            kinit, deltak = d["kinit"], d["deltak"]
            kend = harness.checkkend(kinit, None, deltak, numsoks)
            d["kend"] = kend
            
            #Set script filename and ensure directory exists
            qsubfilename = os.path.join(configuration.QSUBSCRIPTSDIR, "-".join(["full", str(kinit), str(deltak)]) + ".sh")
            ensurepath(qsubfilename) 
            nf = open(qsubfilename, "w")
            
            #Put together info for filenames
            info = "-".join([configuration.foclass.__name__, configuration.POT_FUNC, str(kinit), str(kend), str(deltak), time.strftime("%H%M%S")])
            filename = os.path.join(configuration.RESULTSDIR , "fo-" + info + ".hf5")
            srcdir = os.path.join(configuration.RESULTSDIR, "src-" + info, "")
            ensurepath(filename)
            ensurepath(srcdir)
            d["fofile"] = filename
            d["codedir"] = configuration.CODEDIR
            d["qsublogsdir"] = configuration.QSUBLOGSDIR
            try:
                nf.write(text%d)
            finally:
                nf.close()
    finally:
        f.close()
    return

def main():
    """Process command line options, create qsub scripts and start execution."""
    
    #Parse command line options
    parser = OptionParser()
    parser.add_option("-q", "--quiet",
                  action="store_const", const=logging.FATAL, dest="loglevel", 
                  help="only print fatal error messages")
    parser.add_option("-v", "--verbose",
                  action="store_const", const=logging.INFO, dest="loglevel", 
                  help="print informative messages")
    parser.add_option("--debug",
                  action="store_const", const=logging.DEBUG, dest="loglevel", 
                  help="print lots of debugging information")
        
    (options, args) = parser.parse_args()
    
    # Start logging
    logging.basicConfig(level=options.loglevel)
    
    if os.path.isfile(templatefile):
        genfullscripts(templatefile)
    else:
        print("No template file found at %s!" %templatefile)
        sys.exit(1)
        

if __name__ == "__main__":
    main()
    
    