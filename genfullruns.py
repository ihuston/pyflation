"""genfullruns.py - Generate bash scripts for qsub for multiple first order runs.
Author: Ian Huston
"""
import os
import hconfig
import harness
import time

templatefile = "/home/ith/numerics/fullsubs/full-template.sh"
filestub = "/home/ith/numerics/fullsubs/full-"

sublist = [ {"kinit": 1e-61, "deltak": 1e-61},
            {"kinit": 1e-60, "deltak": 1e-61},
            {"kinit": 1e-59, "deltak": 1e-61},
            {"kinit": 1e-58, "deltak": 1e-61},
            {"kinit": 1e-60, "deltak": 1e-60},
            {"kinit": 1e-59, "deltak": 1e-60},
            {"kinit": 1e-58, "deltak": 1e-60},
            {"kinit": 1e-57, "deltak": 1e-60}]
            
def genfullscripts(tfilename):
    numsoks = hconfig.NUMSOKS
    try:
        f = open(tfilename, "r")
    except IOError:
        raise
    try:
        text = f.read()
        for d in sublist:
            kinit, deltak = d["kinit"], d["deltak"]
            kend = harness.checkkend(kinit, None, deltak, numsoks)
            d["kend"] = kend
            nf = open(filestub + str(kinit) + "-" + str(deltak) + ".sh", "w")
            filename = hconfig.RESULTSDIR + "fo-" + hconfig.foclass.__name__ + "-" + hconfig.POT_FUNC + "-" + str(kinit) + "-" + str(kend) + "-" + str(deltak) + "-" + time.strftime("%H%M%S") + ".hf5"
            harness.ensureresultspath(filename)
            d["fofile"] = filename
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
        print "No template file found!"
        sys.exit(1)