#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N AUCSCalc
#$ -cwd
#$ -l h_rt=12:00:00
#$ -l h_vmem=64G

# Initialise the environment modules
. /etc/profile.d/modules.sh
#Home Directory
HD=/exports/igmm/eddie/marsh-lab/users/snouto
# Job Directory
JD=$HD/paper/aucs_all
PEXEC=/home/s2273299/.conda/envs/features/bin/python
GENES=$HD/paper/dataset/genes3.names.props.csv
MUTATIONS=$HD/vectorized/all.variants.csv.gz
OUTPUT=$JD

$PEXEC $JD/generate_aucs.py --genes=$GENES --mutations=$MUTATIONS --out=$OUTPUT

