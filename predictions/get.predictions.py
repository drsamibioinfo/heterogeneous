import os, sys
import logging
from argparse import ArgumentParser, Namespace
import os, json, joblib as jl
from scipy import stats as st
import glob
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import traceback as tr
import warnings

USED_COLS = ['transcript_count', 'nTPM', 'obs_mis', 'exp_mis', 'oe_mis', 'possible_mis', 'obs_lof', 'mu_lof', 'possible_lof', 'pRec', 'oe_lof', 'oe_lof_lower', 'lof_z', 'oe_lof_upper_rank', 'no_lofs', 'classic_caf', 'max_af', 'no_lofs', 'obs_hom_lof', 'exac_obs_lof', 'exac_exp_lof', 'exac_oe_lof', 'GDI', 'ni', 'clarkd', 'cds_len', 'betweenness', 'interior', 'ionic_n', 'catpi_n', 'aroro_n', 'arosul_n', 'disulphide_n', 'mmhbonds_n', 'mshbonds_n', 'sshbonds_n', 'AAAvgClustering', 'MW', 'rsa_molecule', 'helices_ratio', 'sheets_ratio', 'avgevorate', 'ages', 'rvis', 'n_paralogs', 'Disulfide_bond', 'Formation_of_an_isopeptide_bond', 'cosmis', 'cossyn', 'gnomAD_distance', 'clustering', 'smc_log10_ci_low', 'smc_log10_map', 'is_homomer', 'is_heteromer', 'GO:0005829', 'GO:0005739', 'GO:0000122', 'GO:0006355', 'GO:0006357', 'GO:0032502', 'GO:0032501', 'GO:1901605', 'GO:0006520', 'GO:0019752', 'GO:0044281', 'GO:0140110', 'GO:0001067', 'ndomains', 'max_domain_len', 'evolutionary_rate', 'divergence_score', 'evo_diff_range', 'mean_plddt', 'benign_odds', 'pLI_intolerant', 'exac_pLI_intolerant', 'pLI_tolerant', 'exac_pLI_tolerant', 'protein_len', 'domain_ratio', 'ddG_fold', 'hm_dn_ds', 'localization_CellMembrane', 'localization_Cytoplasm', 'localization_Endoplasmicreticulum', 'localization_Extracellular', 'localization_GolgiApparatus', 'localization_Lysosome/Vacuole', 'localization_Mitochondrion', 'localization_Nucleus', 'localization_Peroxisome', 'localization_Plastid', 'localization_Membranebound', 'plddt_disorder', 'shet', 'efx_raw', 'efx_abs', 'pct_buried', 'ct', 'pDN', 'pGOF', 'pLOF', 'mpm_AlphaMissense', 'mpm_fathmm-XF', 'mpm_CADD', 'mpm_MutationAssessor', 'mpm_MetaRNN', 'mpm_ESM-1v', 'mpm_MVP', 'mpm_DANN', 'mpm_PrimateAI', 'mpm_mutationTCN', 'mpm_REVEL', 'mpm_Eigen', 'mpm_SIFT4G', 'mpm_SIFT', 'mpm_FATHMM', 'mpm_VEST4', 'mpm_MetaSVM', 'mpm_VESPAl', 'mpm_PonP2', 'mpm_LIST-S2', 'mpm_MetaLR', 'mpm_ClinPred', 'mpm_BayesDel', 'mpm_M-CAP', 'mpm_Envision', 'mpm_EVE', 'mpm_ESM-1b', 'mpm_MutPred', 'mpm_MPC', 'mpm_DEOGEN2', 'mpm_PROVEAN', 'MetaSVM_gnomad_mean', 'MetaRNN_gnomad_mean', 'AlphaMissense_gnomad_mean', 'ESM-1v_gnomad_mean', 'MutPred_gnomad_mean', 'SIFT_gnomad_mean', 'EVE_gnomad_mean', 'SIFT4G_gnomad_mean', 'ESM-1b_gnomad_mean', 'DANN_gnomad_mean', 'FATHMM_gnomad_mean', 'DEOGEN2_gnomad_mean', 'fathmm-XF_gnomad_mean', 'PrimateAI_gnomad_mean', 'M-CAP_gnomad_mean', 'REVEL_gnomad_mean', 'MetaLR_gnomad_mean', 'PonP2_gnomad_mean', 'ClinPred_gnomad_mean', 'MutationAssessor_gnomad_mean', 'mutationTCN_gnomad_mean', 'MVP_gnomad_mean', 'BayesDel_gnomad_mean', 'VEST4_gnomad_mean', 'MPC_gnomad_mean', 'Eigen_gnomad_mean', 'VESPAl_gnomad_mean', 'CADD_gnomad_mean', 'LIST-S2_gnomad_mean', 'Envision_gnomad_mean', 'PROVEAN_gnomad_mean']

TOP_FEATURES = [
"GO:0032501",
"ddG_fold",
"GO:0032502",
"gnomAD_distance",
"plddt_disorder",
"shet",
"exac_oe_lof",
"mean_plddt",
"pLOF",
"n_sites"

]

def warn(*args, **kwargs):
    pass


warnings.warn = warn

# Logging
Log_Format = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    filemode="w",
    format=Log_Format,
    level=logging.INFO)

logger = logging.getLogger("RFE Selection")


def rename_redundant_features(df, vep) -> pd.DataFrame:
    # Prepare gnomad Means , rename them as vep_gnomad_mean
    current_vep_feature = f"{vep}_gnomad_mean"
    gnomad_means_cols = [x for x in df.columns if str(x).__contains__("_gnomad_mean")]
    gnomad_means_cols.remove(current_vep_feature)
    temp_df = df.drop(columns=gnomad_means_cols, axis=1)
    temp_df["vep_gnomad_mean"] = temp_df[current_vep_feature].copy()
    temp_df.drop(current_vep_feature, axis=1, inplace=True)
    # prepare mpm_features
    current_vep_feature = f"mpm_{vep}"
    mpm_cols = [x for x in temp_df.columns if str(x).startswith("mpm_")]
    mpm_cols.remove(current_vep_feature)
    temp_df.drop(columns=mpm_cols, axis=1, inplace=True)
    temp_df["mpossible_mutations"] = temp_df[current_vep_feature].copy()
    temp_df.drop(current_vep_feature, axis=1, inplace=True)
    return temp_df


class MultipleModels(object):
    def __init__(self, multiple_models_dir=None):
        self.estimators = []
        self.results = None
        self.m_dir = multiple_models_dir
        self.__load_estimators__()

    def __load_estimators__(self):
        self.estimators.clear()
        gfiles = glob.glob(os.path.join(self.m_dir, "*.m"))
        for file in gfiles:
            model_name, _ = os.path.splitext(os.path.basename(file))
            curr_model = jl.load(file)
            self.estimators.append((model_name, curr_model))

    def predict(self, genes: pd.DataFrame):
        models = []
        results = None
        for name, model in self.estimators:
            prepared_genes = rename_redundant_features(genes, name)
            logger.info(f"Features: {model.feature_names_in_}")
            features = prepared_genes[model.feature_names_in_]
            imputer = SimpleImputer(strategy="constant", fill_value=0.0)
            final_features = imputer.fit_transform(features)
            m_pred = model.predict(final_features)
            m_pred = pd.Series(m_pred)
            models.append(name)
            if results is None:
                results = m_pred
            else:
                results = pd.concat([results, m_pred], axis=1)
        df = pd.DataFrame(results)
        df.columns = models
        df['gene'] = genes['gene'].to_numpy()
        self.results = df
        return df

    def rank(self, ascending=True, pct=True):
        if self.results is None:
            raise Exception("You should call predict method first before ranking genes")
        return self.results.rank(ascending=ascending, axis=0, pct=pct)


def process(args: Namespace):
    genes = pd.read_csv(args.genes, low_memory=False)
    logger.info(f"Removing all genes with empty values in their top 10 important features")
    genes.dropna(subset=TOP_FEATURES,inplace=True)
    logger.info(f"Performing total human proteome predictions.")
    models = MultipleModels(multiple_models_dir=args.multiple)
    predictions = models.predict(genes)
    logger.info(f"Saving the final predictions to: {args.output}")
    out_file = os.path.join(args.output, f"human.proteome.predictions.csv")
    predictions.insert(0,"gene",predictions.pop("gene"))
    predictions.to_csv(out_file, index=False)
    logger.info(f"All Done.")


def main():
    p = ArgumentParser(
        description="This program will produce the final human proteome predictions for all models that exist in the multiple dir.")
    p.add_argument("-m", "--multiple", help="The directory that contains all pickled RF regressor Models.",
                   required=True)
    p.add_argument("-g", "--genes", help="The genes and their collected properties file.", required=True)
    p.add_argument("-o", "--output", help="The output directory where to save the final results.", required=True)
    if len(sys.argv) <= 1:
        p.print_help()
        return
    args = p.parse_args()
    process(args)


if __name__ == '__main__':
    main()

