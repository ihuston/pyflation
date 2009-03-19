#!/bin/sh
#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:10:00
#PBS -N ih-forun-%(kinit)s-%(deltak)s
#PBS -o /home/ith/numerics/qsublogs/forun-%(kinit)s-%(deltak)s.out
#PBS -e /home/ith/numerics/qsublogs/forun-%(kinit)s-%(deltak)s.err
#PBS -V
KINIT=%(kinit)s
DELTAK=%(deltak)s

echo Start: host `hostname`, date `date`
NPROCS=`wc -l < $PBS_NODEFILE`
echo Number of nodes is $NPROCS
echo PBS id is $PBS_JOBID
echo Assigned nodes: `cat $PBS_NODEFILE`

cd /home/ith/numerics

/usr/local/bin/mpiexec --comm=pmi python harness.py -m --kinit $KINIT --deltak $DELTAK


