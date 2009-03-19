#!/bin/sh
#PBS -l nodes=8:ppn=4
#PBS -l walltime=03:00:00
#PBS -N full-%(kinit)s-%(deltak)s
#PBS -o /home/ith/numerics/qsublogs/full-%(kinit)s-%(deltak)s.out
#PBS -e /home/ith/numerics/qsublogs/full-%(kinit)s-%(deltak)s.err
#PBS -V

#Change first order file here
FOFILE=%(fofile)s
KINIT=%(kinit)s
DELTAK=%(deltak)s
KEND=%(kend)s

echo Start: host `hostname`, date `date`
NPROCS=`wc -l < $PBS_NODEFILE`
echo Number of nodes is $NPROCS
echo PBS id is $PBS_JOBID
echo Assigned nodes: `cat $PBS_NODEFILE`

cd /home/ith/numerics

echo Starting first order run:
/usr/local/bin/mpiexec -n 1 --comm=pmi python harness.py -f $FOFILE -m --kinit $KINIT --deltak $DELTAK --kend $KEND
echo First order run complete.
echo Starting second order run:
/usr/local/bin/mpiexec --comm=pmi python harness.py -f $FOFILE -p 
echo Second order run complete.
