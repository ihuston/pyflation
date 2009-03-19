"""genforuns.py - Generate bash scripts for qsub for multiple first order runs.
Author: Ian Huston
"""
import os

templatefile = os.getcwd() + "/forun-template.sh"
filestub = os.getcwd() + "/forun-"
sublist = [ {"kinit": 1e-61, "deltak": 1e-61},
            {"kinit": 1e-60, "deltak": 1e-61},
            {"kinit": 1e-59, "deltak": 1e-61},
            {"kinit": 1e-58, "deltak": 1e-61},
            {"kinit": 1e-60, "deltak": 1e-60},
            {"kinit": 1e-59, "deltak": 1e-60},
            {"kinit": 1e-58, "deltak": 1e-60},
            {"kinit": 1e-57, "deltak": 1e-60}]
            
def genfoscripts(tfilename):
    try:
        f = open(tfilename, "r")
    except IOError:
        raise
    try:
        text = f.read()
        for d in sublist:
            nf = open(filestub + str(d["kinit"]) + "-" + str(d["deltak"]) + ".sh", "w")
            try:
                nf.write(text%d)
            finally:
                nf.close()
    finally:
        f.close()
    return

if __name__ == "__main__":
    if os.path.isfile(templatefile):
        genfoscripts(templatefile)
    else:
        print "No template file found!"
        sys.exit(1)