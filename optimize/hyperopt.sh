#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N OPTIMIZER
#$ -cwd
#$ -l h_rt=120:00:00
#$ -l h_vmem=16G

# Initialise the environment modules
. /etc/profile.d/modules.sh
#Home Directory
HD=/exports/igmm/eddie/marsh-lab/users/snouto
# Job Directory
JD=$HD/paper/optimize_r
OUTDIR=$JD/results
AUCS=$HD/paper/dataset/roc.aucs.csv
GENES=$HD/paper/dataset/genes.final.final.csv
PEXEC=/home/s2273299/.conda/envs/features/bin/python

mkdir -p $OUTDIR

$PEXEC $JD/optimize.py --aucs=$AUCS --genes=$GENES --output=$OUTDIR --vep=$1
