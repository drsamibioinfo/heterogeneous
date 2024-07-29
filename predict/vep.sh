#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N PREDICT
#$ -cwd
#$ -l h_rt=12:00:00
#$ -l h_vmem=32G

# Initialise the environment modules
. /etc/profile.d/modules.sh
#Home Directory
HD=/exports/igmm/eddie/marsh-lab/users/snouto
# Job Directory
JD=$HD/paper/final_predict
DATASET=$HD/paper/dataset
OUTDIR=$JD
VEP=$1
VERBOSE=$2
AUCS=$DATASET/roc.aucs.csv
GENES=$DATASET/genes.final.final.csv
OPTIMAL=$HD/paper/optimize_r/results
NONHOM=$DATASET/non.homologous.set.csv
PEXEC=/home/s2273299/.conda/envs/features/bin/python

mkdir -p $OUTDIR/perfs

$PEXEC $JD/predict.py --aucs=$AUCS --genes=$GENES --optimal=$OPTIMAL --vep=$VEP --nonhom=$NONHOM --output=$OUTDIR --verbose=$VERBOSE
