# -*- coding: utf-8 -*-
"""genfullruns.py - Generate bash scripts for qsub for multiple first order runs.
Author: Ian Huston
"""
import os
import os.path
import hconfig
import harness
import time
import sys
from helpers import ensurepath

templatefile = os.path.join(hconfig.CODEDIR, "full-template.sh")


sublist = [ {"kinit": 0.5e-61, "deltak": 1e-61, "numsoks": 1025},
            {"kinit": 1.5e-61, "deltak": 3e-61, "numsoks": 1025},
            {"kinit": 0.25e-60, "deltak": 1e-60, "numsoks": 1025}]
            
def genfullscripts(tfilename):
    numsoks = hconfig.NUMSOKS
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
            qsubfilename = os.path.join(hconfig.QSUBSCRIPTSDIR, "-".join(["full", str(kinit), str(deltak)]) + ".sh")
            ensurepath(qsubfilename) 
            nf = open(qsubfilename, "w")
            
            #Put together info for filenames
            info = "-".join([hconfig.foclass.__name__, hconfig.POT_FUNC, str(kinit), str(kend), str(deltak), time.strftime("%H%M%S")])
            filename = os.path.join(hconfig.RESULTSDIR , "fo-" + info + ".hf5")
            srcdir = os.path.join(hconfig.RESULTSDIR, "src-" + info, "")
            ensurepath(filename)
            ensurepath(srcdir)
            d["fofile"] = filename
            d["codedir"] = hconfig.CODEDIR
            d["qsublogsdir"] = hconfig.QSUBLOGSDIR
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