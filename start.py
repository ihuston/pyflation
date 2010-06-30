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

if __name__ == "__main__":
    if os.path.isfile(templatefile):
        genfullscripts(templatefile)
    else:
        print("No template file found at %s!" %templatefile)
        sys.exit(1)