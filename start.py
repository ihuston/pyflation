# -*- coding: utf-8 -*-
"""start.py - Generate bash scripts for qsub and execute.
Author: Ian Huston
"""
from __future__ import print_function

import os.path
import configuration
import run_config
import harness
import time
import sys
from helpers import ensurepath
from optparse import OptionParser, OptionGroup
import logging
import subprocess
import helpers


@property
def base_qsub_dict():
    qdict = dict(codedir = run_config.CODEDIR,
                 runname = run_config.PROGRAM_NAME,
                 timelimit = run_config.timelimit,
                 qsublogname = run_config.qsublogname,
                 taskmin = run_config.taskmin,
                 taskmax = run_config.taskmax,
                 hold_jid_list = run_config.hold_jid_list, 
                 fotemplatefile = run_config.fotemplatefile,
                 fulltemplatefile = run_config.fulltemplatefile,
                 foscriptname = run_config.foscriptname,
                 fullscriptname = run_config.fullscriptname               
                 )
    return qdict
    
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
    
    

def main(argv=None):
    """Process command line options, create qsub scripts and start execution."""

    if not argv:
        argv = sys.argv
    
    #Default dictionary for templates
    template_dict = base_qsub_dict
    
    #Parse command line options
    parser = OptionParser()
    
    loggroup = OptionGroup(parser, "Log Options", 
                           "These options affect the verbosity of the log files generated.")
    loggroup.add_option("-q", "--quiet",
                  action="store_const", const=logging.FATAL, dest="loglevel", 
                  help="only print fatal error messages")
    loggroup.add_option("-v", "--verbose",
                  action="store_const", const=logging.INFO, dest="loglevel", 
                  help="print informative messages")
    loggroup.add_option("--debug",
                  action="store_const", const=logging.DEBUG, dest="loglevel", 
                  help="print lots of debugging information",
                  default=run_config.LOGLEVEL)
    parser.add_option_group(loggroup)
    
    cfggroup = OptionGroup(parser, "Simulation configuration Options",
                           "These options affect the options used by the simulation.")
    cfggroup.add_option("--name", action="store", dest="runname", 
                        type="string", help="name of run")
    cfggroup.add_option("--timelimit", action="store", dest="timelimit",
                        type="string", help="time for simulation in format hh:mm:ss")
    cfggroup.add_option("--taskmin", action="store", dest="taskmin",
                        type="string", metavar="NUM", help="minimum task number, default: 1")
    cfggroup.add_option("-t", "--taskmax", action="store", dest="taskmax",
                        type="string", metavar="NUM", help="maximum task number, default: 20")
    parser.add_option_group(cfggroup)
    
    filegrp = OptionGroup(parser, "File options", 
                          "These options override the default choice of template and script files.")
    filegrp.add_option("--fotemplate", action="store", dest="fotemplatefile", 
                       type="string", help="first order template file")
    filegrp.add_option("--fulltemplate", action="store", dest="fulltemplatefile",
                       type="string", help="full program template file")
    filegrp.add_option("--foscript", action="store", dest="foscriptname",
                       type="string", help="first order script name")
    filegrp.add_option("--fullscript", action="store", dest="fullscriptname",
                       type="string", help="full program script name")
    parser.add_option_group(filegrp)
    
    (options, args) = parser.parse_args(args=argv[1:])
    
    if args:
        raise ValueError("No extra command line arguments are allowed!")
    
    #Update dictionary with options
    template_dict.update(options)
    
    helpers.startlogging(log, run_config.logfile, options.loglevel)
    
    #Log options chosen
    log.debug("Generic template dictionary is %s", template_dict)
    
    #First order script creation
    fo_dict = template_dict.copy()
    fo_dict["runname"] += "-fo"
    
    if os.path.isfile(fo_dict["fotemplatefile"]):
        write_out_template(fo_dict["fotemplatefile"],fo_dict["foscriptname"], fo_dict)
    else:
        raise IOError("No template file found at %s!" % fo_dict["fotemplatefile"])

    
    #Launch first order script and get job id
    try:
        fo_jid = launch_qsub(fo_dict["foscriptname"])
        log.info("Submitted first order script with job id %s.", fo_jid)
    except Exception:
        log.error("Error executing script %s", fo_dict["foscriptname"])
        raise
    
    #Write second order file with job_id from first
    full_dict = template_dict.copy()
    full_dict["hold_jid_list"] = fo_jid
    
    if os.path.isfile(full_dict["fulltemplatefile"]):
        write_out_template(full_dict["fulltemplatefile"],full_dict["fullscriptname"], full_dict)
    else:
        raise IOError("No template file found at %s!" % full_dict["fulltemplatefile"])

    
    #Launch full script and get job id
    try:
        full_jid = launch_qsub(full_dict["fullscriptname"])
        log.info("Submitted full script with job id %s.", full_jid)
    except Exception:
        log.error("Error executing script %s", full_dict["fullscriptname"])
        raise
    
    return 0
            

if __name__ == "__main__":
    # Start logging
    log=logging.getLogger()
    try:
        sys.exit(main())
    except Exception as e:
        print("Something went wrong!", file=sys.stderr)
        print(e.message, file=sys.stderr)
        sys.exit(1)
        
    
    