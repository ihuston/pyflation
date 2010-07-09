#!/bin/bash
#$ -l h_rt=%(timelimit)s
#$ -N %(runname)s
#$ -o %(qsublogname)s
#$ -j y
#$ -v PATH,PYTHONPATH,LD_LIBRARY_PATH
#$ -S /bin/bash
%(extra_qsub_params)s

echo -----------------------------------------
echo Start: host `hostname`, date `date`
if [ -n "$SGE_TASK_ID" ]; then
    if [ "$SGE_TASK_ID" != "undefined" ]; then
        echo Task array: $SGE_TASK_FIRST-$SGE_TASK_LAST, step=$SGE_TASK_STEPSIZE
        echo My task-id:$SGE_TASK_ID
    fi
fi

CODEDIR=%(codedir)s

cd $CODEDIR
echo In directory $CODEDIR


FOFILE=%(foresults)s

%(command)s

