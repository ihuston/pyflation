#!/usr/bin/env python
"""start.py - Generate bash scripts for qsub and execute.

"""

#Author: Ian Huston
#For license and copyright information see LICENSE.txt which was distributed with this file.

from __future__ import print_function

import os.path
import sys
from optparse import OptionParser, OptionGroup
import logging
import subprocess
import re

#Try to import run configuration file
try:
    import run_config
except ImportError, e:
    if __name__ == "__main__":
        msg = """Configuration file run_config.py needs to be available."""
        print(msg, e)
        sys.exit(1)
    else:
        raise

try:
    #Local modules from pyflation package
    from pyflation import configuration, helpers
    _debug = configuration._debug
except ImportError,e:
    if __name__ == "__main__":
        msg = """Pyflation module needs to be available. 
Either run this script from the base directory as bin/pyflation_qsubstart.py or add directory enclosing pyflation package to PYTHONPATH."""
        print(msg, e)
        sys.exit(1)
    else:
        raise


                                  

#Dictionary of qsub configuration values
base_qsub_dict = dict(rundir = run_config.RUNDIR,
                 runname = run_config.runname,
                 timelimit = run_config.timelimit,
                 qsublogname = run_config.qsublogname,
                 taskmin = run_config.taskmin,
                 taskmax = run_config.taskmax,
                 hold_jid_list = run_config.hold_jid_list, 
                 templatefile = run_config.templatefile,
                 foscriptname = run_config.foscriptname,
                 srcscriptname = run_config.srcscriptname,
                 src_indivscriptname = run_config.src_indivscriptname,
                 mrgscriptname = run_config.mrgscriptname,
                 soscriptname = run_config.soscriptname,
                 cmbscriptname = run_config.cmbscriptname,
                 foresults = run_config.foresults,
                 srcstub = run_config.srcstub,
                 extra_qsub_params = "",
                 command = "",      
                 )
    
def launch_qsub(qsubscript):
    """Submit the job to the queueing system using qsub.
    
    Return job id of new job.
    """
    qsubcommand = ["qsub", "-terse", qsubscript]
    try:
        newprocess = subprocess.Popen(qsubcommand, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        raise
    
    result = newprocess.stdout.read()
    error_msg = newprocess.stderr.read()
    
    if error_msg:
        log.error("Error executing script %s", qsubscript)
        raise Exception(error_msg)
    # Get job id
    job_id = re.search(r"(\d+).?\S*", result).groups()[0]
    #Log job id info
    if _debug:
        log.debug("Submitted qsub script %s with job id %s.", qsubscript, job_id)
    
    return job_id

def write_out_template(templatefile, newfile, textdict):
    """Write the textdict dictionary using the templatefile to newfile."""
    if not os.path.isfile(templatefile):
        raise IOError("Template file %s does not exist!" % templatefile)
    try:
        f = open(templatefile, "r")
        text = f.read()
    except IOError:
        raise
    finally:
        f.close()
    
    if os.path.isfile(newfile):
        raise IOError("File %s already exists! Please delete or use another filename." % newfile)
    #Ensure directory exists for new file
    try:
        helpers.ensurepath(newfile)
    except IOError:
        raise
    try: 
        nf = open(newfile, "w")
        nf.write(text%textdict)
    except IOError:
        raise
    finally:
        nf.close()
    return
    
def first_order_dict(template_dict):
    """Return dictionary for first order qsub script.
    Copies template_dict so as not to change values."""
    fo_dict = template_dict.copy()
    fo_dict["runname"] += "-fo"
    fo_dict["qsublogname"] += "-fo"
    fo_dict["command"] = "python bin/pyflation_firstorder.py"
    return fo_dict

def source_dict(template_dict, fo_jid=None):
    """Return dictionary for source qsub script."""
    #Write second order file with job_id from first
    src_dict = template_dict.copy()
    src_dict["hold_jid_list"] = fo_jid
    src_dict["runname"] += "-src"
    src_dict["qsublogname"] += "-node-$TASK_ID"
    src_dict["extra_qsub_params"] = ("#$ -t " + src_dict["taskmin"] + "-" +
                                    src_dict["taskmax"] +"\n#$ -hold_jid " + 
                                    src_dict["hold_jid_list"] +
                                    "\n#$ -r y")
    #Formulate source term command
    src_dict["command"] = ("python bin/pyflation_source.py --taskmin=$SGE_TASK_FIRST "
                           "--taskmax=$SGE_TASK_LAST --taskstep=$SGE_TASK_STEPSIZE "
                           "--taskid=$SGE_TASK_ID  --overwrite")
    return src_dict

def source_indiv_dict(template_dict):
    """Return dictionary for source qsub script."""
    #Write second order file with job_id from first
    src_dict = template_dict.copy()
    src_dict["hold_jid_list"] = ""
    src_dict["runname"] += "-src-indiv"
    src_dict["qsublogname"] += "-node-$1"
    src_dict["extra_qsub_params"] = "\n".join(["SGE_TASK_FIRST=" + src_dict["taskmin"],
                                              "SGE_TASK_LAST=" + src_dict["taskmax"],
                                              "SGE_TASK_STEPSIZE=1",
                                              "SGE_TASK_ID=$1"])
    #Formulate source term command
    src_dict["command"] = ("python bin/pyflation_source.py --taskmin=$SGE_TASK_FIRST "
                           "--taskmax=$SGE_TASK_LAST --taskstep=$SGE_TASK_STEPSIZE "
                           "--taskid=$SGE_TASK_ID  --overwrite")
    return src_dict

def merge_dict(template_dict, src_jid=None):
    """Return dictionary for first order qsub script.
    Copies template_dict so as not to change values."""
    mrg_dict = template_dict.copy()
    mrg_dict["runname"] += "-mrg"
    mrg_dict["hold_jid_list"] = src_jid
    mrg_dict["qsublogname"] += "-mrg"
    mrg_dict["extra_qsub_params"] = ("#$ -hold_jid " + mrg_dict["hold_jid_list"])
    mrg_dict["command"] = "python bin/pyflation_srcmerge.py --merge"
    return mrg_dict

def second_order_dict(template_dict, mrg_jid=None):
    """Return dictionary for first order qsub script.
    Copies template_dict so as not to change values."""
    so_dict = template_dict.copy()
    so_dict["runname"] += "-so"
    so_dict["hold_jid_list"] = mrg_jid
    so_dict["qsublogname"] += "-so"
    so_dict["extra_qsub_params"] = ("#$ -hold_jid " + so_dict["hold_jid_list"])
    so_dict["command"] = "python bin/pyflation_secondorder.py"
    return so_dict

def combine_dict(template_dict, so_jid=None):
    """Return dictionary for first order qsub script.
    Copies template_dict so as not to change values."""
    cmb_dict = template_dict.copy()
    cmb_dict["runname"] += "-cmb"
    cmb_dict["hold_jid_list"] = so_jid
    cmb_dict["qsublogname"] += "-cmb"
    cmb_dict["extra_qsub_params"] = ("#$ -hold_jid " + cmb_dict["hold_jid_list"])
    cmb_dict["command"] = "python bin/pyflation_combine.py"
    return cmb_dict

def main(argv=None):
    """Process command line options, create qsub scripts and start execution."""

    if not argv:
        argv = sys.argv
    
    #Default dictionary for templates
    template_dict = base_qsub_dict.copy()
    
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
    filegrp.add_option("--template", action="store", dest="templatefile", 
                       type="string", help="qsub template file")
    filegrp.add_option("--foscript", action="store", dest="foscriptname",
                       type="string", help="first order script name")
    filegrp.add_option("--srcscript", action="store", dest="srcscriptname",
                       type="string", help="source integration script name")
    parser.add_option_group(filegrp)
    
    (options, args) = parser.parse_args(args=argv[1:])
    
    if args:
        raise ValueError("No extra command line arguments are allowed!")
    
    #Update dictionary with options
    for key in template_dict.keys():
        if getattr(options, key, None):
            template_dict[key] = getattr(options, key, None)
    
    #Find templatefile
    if options.templatefile and os.path.isfile(options.templatefile):
        goodtemplatefile = options.templatefile
    elif os.path.isfile(run_config.templatefile):
        goodtemplatefile = run_config.templatefile
    elif os.path.isfile(os.path.join([os.path.abspath(run_config.__file__), 
                                      run_config.templatefilename])):
        goodtemplatefile = os.path.join([os.path.abspath(run_config.__file__), 
                                      run_config.templatefilename])
    elif os.path.isfile(os.path.join([os.getcwd(), run_config.templatefilename])):
        goodtemplatefile = os.path.join([os.getcwd(), run_config.templatefilename])
    else:
        #Can't find templatefile
        raise IOError("Can't find the qsub template file named %s." % run_config.templatefilename)
    template_dict["templatefile"] = goodtemplatefile
    
    
    helpers.startlogging(log, run_config.logfile, options.loglevel)
    
    #Log options chosen
    log.debug("Generic template dictionary is %s", template_dict)
    
    #First order calculation
    #First order script creation
    fo_dict = first_order_dict(template_dict)    
    write_out_template(fo_dict["templatefile"],fo_dict["foscriptname"], fo_dict)
    #Launch first order script and get job id
    fo_jid = launch_qsub(fo_dict["foscriptname"])
  
    #Source term calculation
    src_dict = source_dict(template_dict, fo_jid=fo_jid)
    write_out_template(src_dict["templatefile"],src_dict["srcscriptname"], src_dict)
    #Launch full script and get job id
    src_jid = launch_qsub(src_dict["srcscriptname"])
    
    #Merger of source terms and combining with firstorder results
    mrg_dict = merge_dict(template_dict, src_jid=src_jid)
    write_out_template(mrg_dict["templatefile"], mrg_dict["mrgscriptname"], mrg_dict)
    #Launch full script and get job id
    mrg_jid = launch_qsub(mrg_dict["mrgscriptname"])
    
    #Second order calculation
    so_dict = second_order_dict(template_dict, mrg_jid=mrg_jid)
    write_out_template(so_dict["templatefile"], so_dict["soscriptname"], so_dict)
    #Launch full script and get job id
    so_jid = launch_qsub(so_dict["soscriptname"])
    
    #Combination of final results
    cmb_dict = combine_dict(template_dict, so_jid=so_jid)
    write_out_template(cmb_dict["templatefile"], cmb_dict["cmbscriptname"], cmb_dict)
    #Launch full script and get job id
    cmb_jid = launch_qsub(cmb_dict["cmbscriptname"])
        
    #Write out individual source term qsub file
    src_indiv_dict = source_indiv_dict(template_dict)
    write_out_template(src_indiv_dict["templatefile"],src_indiv_dict["src_indivscriptname"], src_indiv_dict)
    
    return 0
            

if __name__ == "__main__":
    # Start logging
    log=logging.getLogger()
    try:
        sys.exit(main())
    except Exception as e:
        log.exception("Something went wrong!")
        sys.exit(1)
        
    
    