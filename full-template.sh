#!/bin/bash
#$ -l h_rt=%(timelimit)s
#$ -N %(runname)s
#$ -o %(qsublogname)s
#$ -j y
#$ -v PATH,PYTHONPATH,LD_LIBRARY_PATH
#$ -S /bin/bash
#$ -t %(taskmin)s-%(taskmax)s
#$ -hold_jid %(hold_jid_list)s

echo -----------------------------------------
echo Start: host `hostname`, date `date`
echo Task array: {$SGE_TASK_FIRST}-{$SGE_TASK_LAST}, step={$SGE_TASK_STEPSIZE}
echo My task-id:{$SGE_TASK_ID}

#Change first order file here
FOFILE=%(foresults)s
CODEDIR=%(codedir)s

cd $CODEDIR
echo In directory $CODEDIR

echo Do some work here now!
echo Source filename %(srcstub)s{$SGE_TASK_ID}.hf5