#!/bin/bash
VEPS="AlphaMissense fathmm-XF CADD Polyphen2_HumVar VARITY_ER MutationAssessor MetaRNN ESM-1v MVP DANN PrimateAI mutationTCN VARITY_R REVEL Eigen SIFT4G SIFT FATHMM VEST4 MetaSVM VESPAl PonP2 LIST-S2 MetaLR ClinPred BayesDel M-CAP Polyphen2_HumDiv Envision EVE ESM-1b MutPred MPC DEOGEN2 PROVEAN"

for vep in $VEPS; do
  qsub vep.sh $vep
done
