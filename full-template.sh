#!/bin/bash
#$ -pe orte 7
#$ -l h_rt=03:00:00
#$ -N full-%(kinit)s-%(deltak)s
#$ -o %(qsublogsdir)s/full-%(kinit)s-%(deltak)s.out
#$ -e %(qsublogsdir)s/full-%(kinit)s-%(deltak)s.err
#$ -v PATH,PYTHONPATH,LD_LIBRARY_PATH
#$ -S /bin/bash

#Change first order file here
FOFILE=%(fofile)s
KINIT=%(kinit)s
DELTAK=%(deltak)s
KEND=%(kend)s
CODEDIR=%(codedir)s

echo -----------------------------------------
echo Start: host `hostname`, date `date`
echo NSLOTS: $NSLOTS
declare -i TOTNUMPROCS
TOTNUMPROCS=4*$NSLOTS
echo TOTNUMPROCS: $TOTNUMPROCS

cd $CODEDIR
echo In directory $CODEDIR

echo Starting first order run:
mpirun -np 1 python harness.py -f $FOFILE -m --kinit $KINIT --deltak $DELTAK --kend $KEND
echo First order run complete.
echo Starting second order run:
mpirun -np $TOTNUMPROCS python harness.py -f $FOFILE -p 
echo Second order run complete.
