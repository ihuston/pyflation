#!/bin/bash
#$ -l h_rt=%(timelimit)s
#$ -N %(runname)s
#$ -o %(qsublogname)s
#$ -j y
#$ -v PATH,PYTHONPATH,LD_LIBRARY_PATH
#$ -S /bin/bash

echo -----------------------------------------
echo Start: host `hostname`, date `date`

echo Do some work here now!

echo Filename %(foresults)s

