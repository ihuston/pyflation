"""gensoruns.py - Generate bash scripts for qsub for multiple second order runs.
Author: Ian Huston
"""
import os
import re


sorundir = "/home/ith/numerics/sosubs"
templatefile = sorundir + "/so-template.sh"

def gensrcscripts(tfilename, dirname=None):
    """Generate qsub bash scripts for second order run from first order files."""
    if not dirname:
        dirname = os.getcwd()
    filestub = sorundir + "/so-"
    try:
        tf = open(tfilename, "r")
    except IOError:
        raise
    regex = re.compile("([0-9.e]*-[0-9]{1,3})-")
    try:
        text = tf.read()
        sublist = [{"fofile":os.path.join(dirname, f), "kinit":regex.findall(f)[0], "deltak":regex.findall(f)[2]} for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f)) and os.path.splitext(f)[1] == ".hf5" and "foandsrc" in f[:8]]
        for d in sublist:
            nf = open(filestub + str(d["kinit"]) + "-" + str(d["deltak"]) + ".sh", "w")
            try:
                nf.write(text%d)
            finally:
                nf.close()
    finally:
        tf.close()
    return

if __name__ == "__main__":
    if os.path.isfile(templatefile):
        gensrcscripts(templatefile)
    else:
        print "No template file found!"
        sys.exit(1)