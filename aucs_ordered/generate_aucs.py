import os, sys
import numpy as np
import pandas as pd
import logging
import traceback as tr
from argparse import ArgumentParser
from pandas.api.types import is_numeric_dtype
import traceback as tr
from collections import defaultdict

# Logging
Log_Format = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    filemode="w",
    format=Log_Format,
    level=logging.INFO)

logger = logging.getLogger("ROC-AUC Generation Process")

VEPS = ["UNEECON", 'AlphaMissense', "PonP2", "Grantham", "EVE", "GenoCanyon", "VARITY_ER", "fathmm-XF", "PrimateAI",
        "MOIpred_benign",
        "PROVEAN", "CONDEL", "sequence_unet", "BayesDel", "SNAP2", "CAPICE", "MOIpred_dominant", "MPC", "phyloP",
        "MOIpred_recesive", "Eigen-pred", "MetaLR", "VEST4", "MutationTaster", "Tranception", "M-CAP", "LRT",
        "fathmm-MKL", "SiPhy", "LIST-S2", "ClinPred", "SIFT", "VESPAl", "DeepSAV", "MetaRNN", "SIFT4G", "LINSIGHT",
        "DeMaSk", "PAPI", "CPT", "REVEL", "PonPS", "Polyphen2_HumDiv", "phastCons", "ESM-1b", "DeepSequence", "InMeRF",
        "BLOSUM62", "CADD", "MISTIC", "GERP++", "COSMIS", "gMVP", "NetDiseaseSNP", "MetaSVM", "MVP", "MutPred",
        "SuSPect", "Polyphen2_HumVar", "FATHMM", "DANN", "Envision", "ESM-1v", "VARITY_R", "Rhapsody", "fitCons",
        "EVmutation_epistatic", "MutationAssessor", "Eigen", "mutationTCN", "EVmutation_independent", "DEOGEN2"]

inverted_veps = ["SIFT", "SIFT4G", "PROVEAN", "BLOSUM62", "FATHMM", "DeepSequence", "Envision", "LRT", "EVcouplings",
                 "LRT",
                 "MOIpred_benign", "mutationTCN", "Tranception",
                 "ESM-1v", "MTBAN", "EVcouplings_epistatic", "EVcouplings_independent",
                 "BLOSUM62", "ESM-1b", "COSMIS", "DeMaSk", "EVmutation_epistatic", "EVmutation_independent"]


def is_inverted(vep):
    try:
        res = inverted_veps.index(vep) >= 0
        if res == True:
            logger.info(f"INVERTED: {vep}")
        return res
    except ValueError as e:
        return False


def calculate_avg_auc(args, data, col='REVEL') -> pd.DataFrame:
    vep_file = os.path.join(args.out, "aucs", f"{col}.aucs.csv")
    uniclass_file = os.path.join(args.out, "uniclass", f"{col}.csv")
    if os.path.exists(vep_file):
        return pd.read_csv(vep_file)
    os.makedirs(os.path.join(args.out, "aucs"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "uniclass"), exist_ok=True)
    from sklearn import metrics
    # get unique genes
    genes = np.unique(data['gene'].astype(str))
    scores = []
    gene_list = []
    errors = 0
    uniclass = set([])

    col_isinverted = is_inverted(col)

    if col_isinverted:
        logger.info(f"Inverting: {col}")

    for gene in genes:
        sel_mask = data['gene'] == gene
        if args.pathogenic_only:
            logger.info(f"Calculating AUC based only on Pathogenic Mutations.")
            sel_mask = sel_mask & (data['significance'] == 1)
        Y_true = data.loc[sel_mask, "significance"]
        Y_pred = data.loc[sel_mask, col]

        if col_isinverted:
            Y_pred = 1 - Y_pred

        try:
            auc = metrics.roc_auc_score(Y_true, Y_pred)
            gene_list.append(gene)
            scores.append(auc)
        except ValueError as e:
            uniclass.add(gene)
            errors += 1
    logger.info(f"Total Genes with uniclass significance : {errors}")
    df_genes = pd.DataFrame({"gene": gene_list, "scores": scores})
    df_genes.to_csv(vep_file, index=False)
    with open(uniclass_file, "w") as writer:
        uniclasses = list(uniclass)
        uniclasses = "\n".join([str(x) for x in uniclasses])
        writer.write(uniclasses)
    logger.info(f"Done writing the uniclass genes for : {col}")
    return df_genes


def get_excluded_columns(df):
    numeric_columns = [x for x in df.columns if (
            is_numeric_dtype(df[x].dtype) and x not in ["start_position", "end_position",
                                                        "entrezgene_id"])]
    # binary_cols = []
    # near_zero_variance = []
    # for col in numeric_columns:
    #     col_data = df[col].unique().tolist()
    #     col_data = [x for x in col_data if not np.isnan(x)]
    #     if len(col_data) > 2:
    #         continue
    #     if len(col_data) == 2:
    #         if all([(col_data[0] in [0, 1]), (col_data[1] in [0, 1])]):
    #             binary_cols.append(col)
    #         else:
    #             near_zero_variance.append(col)
    #     else:
    #         near_zero_variance.append(col)

    excluded_columns = [x for x in df.columns if "q" in x and x != "Ubiquitination"]
    # Removing all Binary Columns
    # excluded_columns += [x for x in df.columns if "GO:" in x]
    # excluded_columns += binary_cols
    # excluded_columns += near_zero_variance

    return excluded_columns


def filter_variants(args, variants):
    pathogenic_set = variants[variants['significance'] == 1].groupby('uniprot_id').size().reset_index(name='counts')
    pathogenic_set = pathogenic_set[pathogenic_set['counts'] >= args.minvariants]
    benign_set = variants[variants['significance'] == 0].groupby('uniprot_id').size().reset_index(name='counts')
    benign_set = benign_set[benign_set['counts'] >= args.minvariants]
    df = pd.merge(pathogenic_set, benign_set, how='inner', on='uniprot_id')
    return df['uniprot_id'].tolist()


def change_to_numeric(num):
    try:
        return float(num)
    except Exception as e:
        logger.error(str(e))
        return 0.0


def process(args) -> pd.DataFrame:
    logger.info("Started Calculating ROC-AUC of different VEPs across human protein coding genes")
    ofile = os.path.join(args.out, "all.results.csv")
    genes = pd.read_csv(args.genes, low_memory=False)
    genes.loc[np.isinf(genes['benign_odds']), "benign_odds"] = 0
    gzipped_ext = str(args.mutations).endswith(".gz")
    if gzipped_ext:
        mutations = pd.read_csv(args.mutations, low_memory=False, compression="gzip")
    else:
        mutations = pd.read_csv(args.mutations, low_memory=False)
    logger.info("Making sure all veps are numeric.")
    for vep in VEPS:
        mutations[vep] = mutations[vep].apply(change_to_numeric)
    logger.info("Done.")
    uniprots = filter_variants(args, mutations)
    filtered_mutations = mutations[mutations['uniprot_id'].isin(uniprots)]
    filtered_mutations = pd.merge(filtered_mutations, genes[['gene', 'uniprot_id']],
                                  how='left', on='uniprot_id')
    filtered_mutations.drop('uniprot_id', axis=1, inplace=True)
    if args.ordered_only:
        logger.info("Restricting analysis to only variants within highly ordered regions.....")
        ordered_only = filtered_mutations['score'] > 50
        filtered_mutations = filtered_mutations.loc[ordered_only, :]
    genes_labels = genes['gene'].copy().tolist()
    total_aucs = {
        "gene": genes_labels
    }
    for VEP in VEPS:
        try:
            logger.info(f"Working on VEP: {VEP}")
            cleaned_mutations = filtered_mutations.dropna(subset=[VEP], axis=0)
            aucs = calculate_avg_auc(args, cleaned_mutations, col=VEP)
            current_values = [None] * len(genes_labels)
            for idx, row in aucs.iterrows():
                gene, score = row['gene'], row['scores']
                gene_idx = genes_labels.index(gene)
                current_values[gene_idx] = score
            total_aucs[VEP] = current_values
        except Exception as e:
            tr.print_exc()
            continue
    logger.info("Done calculating all VEPS ROC-AUCs.")
    final_df = pd.DataFrame(total_aucs)
    logger.info("Saving the final DF of ROC-AUCs")
    outfile = os.path.join(args.out, "roc.aucs.csv")
    final_df.dropna(subset=VEPS, inplace=True, how='all')
    final_df.to_csv(outfile, index=False)
    logger.info("Done saving the ROC-AUC file.")


def main():
    p = ArgumentParser(description="This program will calculate ROC-AUC for Different VEPs for different human "
                                   "protein-coding "
                                   "genes.")
    p.add_argument("-g", "--genes",
                   help="The data file which contains all genes identifiers along with all features sets.",
                   required=True)
    p.add_argument("-m", "--mutations", help="Mutation file", required=True)
    p.add_argument("-d", "--dir",
                   help="Directory where Protein files containing VEPs scores are found", required=False)
    p.add_argument("-t", "--minvariants",
                   help="Minimum number of variants to filter out genes having lower variants count "
                        "than this number. Defaults: 10", type=int, default=10)
    p.add_argument("-l", "--ordered-only", const=True, nargs="?", default=False,
                   help="Whether to restrict analysis only "
                        "to ordered variants, "
                        "Defaults: False")
    p.add_argument("-pa", "--pathogenic-only", const=True, nargs="?",
                   default=False, help="Whether to calculate AUC per gene based only on pathogenic mutations")
    p.add_argument("-p", "--permutations", type=int, default=5, help="The number of permutation rounds to test "
                                                                     "pretrained model's feature importances. Defaults: 5")
    p.add_argument("-o", "--out", help="Full Output Directory to save the results", required=True)

    if len(sys.argv) <= 1:
        p.print_help()
        return
    args = p.parse_args()
    process(args)


if __name__ == '__main__':
    main()

