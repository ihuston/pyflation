# -*- coding: utf-8 -*-
"""start.py - Generate bash scripts for qsub and execute.
Author: Ian Huston
"""
import os.path
import configuration
import run_config
import harness
import time
import sys
from helpers import ensurepath
from optparse import OptionParser
import logging
import subprocess

fotemplatefile = os.path.join(configuration.CODEDIR, "forun-template.sh")
fulltemplatefile = os.path.join(configuration.CODEDIR, "full-template.sh")

        
def genfullscript(tfilename):
    numsoks = configuration.NUMSOKS
    #Put k values in dictionary
    kinit, deltak = d["kinit"], d["deltak"]
    kend = harness.checkkend(kinit, deltak, numsoks)
    d["kend"] = kend
    
    #Set script filename and ensure directory exists
    qsubfilename = os.path.join(configuration.QSUBSCRIPTSDIR, "-".join(["full", str(kinit), str(deltak)]) + ".sh")
    ensurepath(qsubfilename) 
            
    #Put together info for filenames
    info = "-".join([configuration.foclass.__name__, configuration.POT_FUNC, str(kinit), str(kend), str(deltak), time.strftime("%H%M%S")])
    filename = os.path.join(configuration.RESULTSDIR , "fo-" + info + ".hf5")
    srcdir = os.path.join(configuration.RESULTSDIR, "src-" + info, "")
    ensurepath(filename)
    ensurepath(srcdir)
    d["fofile"] = filename
    d["codedir"] = configuration.CODEDIR
    d["qsublogsdir"] = configuration.QSUBLOGSDIR
        
        write_out_template(tfilename, qsubfilename, d)
    return

def launch_qsub(qsubscript):
    """Submit the job to the queueing system using qsub.
    
    Return job id of new job.
    """
    qsubcommand = ["qsub", "-terse", qsubscript]
    newprocess = subprocess.Popen(qsubcommand, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    result = newprocess.stdout.read()
    error_msg = newprocess.stderr.read()
    
    if error_msg:
        raise Exception(error_msg)
    
    job_id = result.rstrip()
    
    return job_id

def write_out_template(templatefile, newfile, textdict):
    """Write the textdict dictionary using the templatefile to newfile."""
    try:
        f = open(templatefile, "r")
        text = f.read()
    except IOError:
        raise
    finally:
        f.close()
        
    #Ensure directory exists for new file
    try:
        ensurepath(newfile)
    except IOError:
        raise
    try: 
        nf = open(newfile, "w")
        nf.write(text%textdict)
    except IOError:
        raise
    finally:
        nf.close()
    
    

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
    
    