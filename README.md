# Understanding the heterogeneous performance of variant effect predictors across human protein-coding genes paper

This repository contains all the scripts needed to reproduce the results discussed in the paper titled "Understanding the heterogeneous performance of variant effect predictors across human protein-coding genes"

- Each folder contains code that computes results for specific stage in the whole pipeline

# Contents

Each folder contains linux bash script which is used as job to submit the python script to the Sun Grid Engine (SGE , Cluster manager)

| Folder | Description |
| --- | --- |
| aucs_all | The python script calculates AUROC for each human protein-coding genes for each VEP |
| aucs_ordered | this folder contains scripts to calculate the AUROC for the human protein-coding genes excluding the disordered residues |
| optimize | this folder contains scripts to perform hyper-parameter optimization using Quasi Monte carlo Sampling strategy for 300 trials using Optuna framework for each VEP separately |
| predict | this folder contains the training logic that trains RF model per each VEP separately utilizing the optimal hyperparameters found in the previous optimize stage |
| predictions | this folder uses the trained RF models per each VEP to predict AUROC for all human protein-coding genes |







