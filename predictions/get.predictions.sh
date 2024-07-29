#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N PROTEOME
#$ -cwd
#$ -l h_rt=8:00:00
#$ -l h_vmem=32G

# Initialise the environment modules
. /etc/profile.d/modules.sh
#Home Directory
HD=/exports/igmm/eddie/marsh-lab/users/snouto
PEXEC=/home/s2273299/.conda/envs/features/bin/python
# Job Directory
JD=$HD/paper/final_predictions
DATASET=$HD/paper/dataset
GENES=$DATASET/genes.final.final.csv
MDIR=/exports/igmm/eddie/marsh-lab/users/snouto/paper/final_predict/multiple
OUTPUT=$JD


$PEXEC $JD/get.predictions.py --genes=$GENES --multiple=$MDIR --output=$OUTPUT


