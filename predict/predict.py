import os, sys
import logging
from argparse import ArgumentParser
import os, json, joblib as jl
from scipy import stats as st
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from pandas.api.types import is_numeric_dtype
from sklearn import metrics as m
from sklearn.model_selection import train_test_split
import traceback as tr

import warnings


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

multiple_dir = None


class MultipleModels(object):
    def __init__(self, multiple_models_dir=None):
        self.estimators = []
        self.results = None
        if multiple_models_dir is None:
            self.m_dir = multiple_dir
        else:
            self.m_dir = multiple_models_dir
        self.__load_estimators__()

    def __load_estimators__(self):
        self.estimators.clear()
        gfiles = glob.glob(os.path.join(self.m_dir, "*.m"))
        for file in gfiles:
            model_name, _ = os.path.splitext(os.path.basename(file))
            curr_model = jl.load(file)
            self.estimators.append((model_name, curr_model))

    def predict(self, features: pd.DataFrame, imp_standardise=False):
        final_features = features
        if imp_standardise:
            pip = make_pipeline(SimpleImputer(strategy="constant",fill_value=0.0))
            final_features = pip.fit_transform(features)
        models = []
        results = None
        for name, model in self.estimators:
            m_pred = model.predict(final_features)
            m_pred = pd.Series(m_pred)
            models.append(name)
            if results is None:
                results = m_pred
            else:
                results = pd.concat([results, m_pred], axis=1)
        df = pd.DataFrame(results)
        df.columns = models
        self.results = df
        return df

    def rank(self, ascending=True, pct=True):
        if self.results is None:
            raise Exception("You should call predict method first before ranking genes")
        return self.results.rank(ascending=ascending, axis=0, pct=pct)


def get_params(data):
    return {"n_estimators": 5000}


def kl_divergence(p, q):
    """
     This function will calculate Kullback-Leibler Divergence Score
    :param p: The continuous first distribution
    :param q: the continuous second distribution
    :return: KL Score
    """
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def evaluate(args, genes, proteome, imp_standardize=False):
    logger.info("Started the evaluation process")
    model = MultipleModels(multiple_models_dir=multiple_dir)
    y_pred = model.predict(proteome, imp_standardise=imp_standardize)
    ranking = model.rank(pct=True, ascending=False)
    df1 = pd.concat([genes, y_pred], axis=1)
    df2 = pd.concat([genes, ranking], axis=1)
    logger.info("Saving human proteome predictions and percentile ranking.")
    df1.to_csv(os.path.join(args.output, "human.proteome.predictions.csv"), index=False)
    df2.to_csv(os.path.join(args.output, "human.proteome.percentiles.csv"), index=False)


def get_metrics(model, data, vep):
    # 1-(1-R2)*(n-1)/(n-p-1)
    all_y, all_x = data.pop(vep), data
    y_pred = model.predict(all_x)
    r, _ = st.spearmanr(all_y, y_pred)
    r2 = m.r2_score(all_y, y_pred)
    ev = m.explained_variance_score(all_y, y_pred)
    rmse = np.sqrt(m.mean_squared_error(all_y, y_pred))
    return r, r2, ev, rmse


def get_optimal_params(args, vep):
    try:

        f = os.path.join(args.optimal, f"{vep}.params.txt")
        with open(f, "r") as reader:
            contents = reader.read()
        return json.loads(contents)

    except Exception as e:
        logger.error(str(e))
        return None


def save_testing_genes(args, vep_name, alldata: pd.DataFrame, x_test: pd.DataFrame):
    try:
        out_dir = os.path.join(args.output, "testing")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{vep_name}.csv")
        testing_genes = alldata.iloc[x_test.index]
        testing_genes[['gene']].to_csv(out_file, index=False)
    except Exception as e:
        logger.error(str(e))


def save_important_features(args, features, importances, vep_name, folder_name="features"):
    try:
        out_folder = os.path.join(args.output, folder_name)
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
        out_features_file = os.path.join(out_folder, f"{vep_name}.features.csv")
        features_scores = list(zip(features, importances))
        df = pd.DataFrame(features_scores, columns=["features", "scores"])
        df['vep_name'] = vep_name
        logger.info(f"Saving Features File to : {out_features_file}")
        df.to_csv(out_features_file, index=False)
        return df
    except Exception as e:
        tr.print_exc()
        return None


def train_the_final_model(args, vep, data, multiple_dir):
    logger.info(f"Training the final Model for VEP: {vep}")
    y, x = data.pop(vep), data
    y = y.round(decimals=2)
    best_params = get_optimal_params(args, vep)
    default_params = dict(n_estimators=2000, max_depth=30, min_samples_split=6)
    if best_params:
        best_params['max_depth'] = int(round(best_params['max_depth']))
        best_params['n_estimators'] = int(round(best_params['n_estimators']))
        best_params["min_samples_split"] = int(round(best_params["min_samples_split"]))
        model = RandomForestRegressor(**best_params)
    else:
        model = RandomForestRegressor(**default_params)
    model.fit(x, y)
    model_name = f"{vep}.m"
    jl.dump(model, os.path.join(multiple_dir, model_name))
    logger.info(f"Extracting {vep} Model features..")
    save_important_features(args, x.columns, model.feature_importances_, vep)


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


todrop = "rvis_percentile,hap_prop,AAinteractions_ratio,smc_log10_ci_high,sasa_molecule,mu_mis,mis_z,GO:0043436,GO:0006082,oe_mis_lower,end_position,hydrophobic_n,obs_het_lof,exp_hom_lof,domain_len_mean,closeness,eigen,oe_lof_upper_bin,degree,ss_disorder,GDI.Phred,pNull,gnomad_vars_percent,GO:0048856,surface,oe_mis_upper,oe_lof_upper,dcentrality,GO:2001141,localization_Soluble,GO:1903506,exp_lof,other_distance,ns_score".split(
    ",")


def process(args):
    global multiple_dir
    multiple_dir = os.path.join(args.output, "multiple")
    folds_dir = os.path.join(args.output, "folds")
    os.makedirs(multiple_dir, exist_ok=True)
    os.makedirs(folds_dir, exist_ok=True)
    genes = pd.read_csv(args.genes)
    selected_columns = genes.columns[~genes.columns.isin(todrop)]
    genes = genes[selected_columns]
    aucs = pd.read_csv(args.aucs)
    nonhom = pd.read_csv(args.nonhom)
    nonhom_genes = nonhom["gene"]
    logger.info("Filtering only for non-homologous set of genes....")
    genes_veps = aucs[aucs['gene'].isin(nonhom_genes)]
    genes.drop_duplicates(subset=['gene'], inplace=True)
    genes = rename_redundant_features(genes, args.vep)
    numeric_columns = [x for x in genes.columns if (
            is_numeric_dtype(genes[x].dtype) and x not in ["start_position", "end_position",
                                                           "entrezgene_id"])]
    performances = []
    default_params = dict(n_estimators=2000, max_depth=30, min_samples_split=6)
    vep = args.vep
    logger.info(f"Working on: {vep}")
    target_column = vep
    nonhomologous_data = pd.merge(genes_veps[["gene", target_column]], genes[numeric_columns + ["gene"]], how='left',
                                  on='gene')
    all_data = pd.merge(aucs[['gene', target_column]], genes[numeric_columns + ["gene"]], how='left', on="gene")
    best_params = get_optimal_params(args, vep)
    for idx in range(100):
        try:
            all_cols = [target_column] + numeric_columns
            vep_scores = nonhomologous_data[target_column]
            imputer = SimpleImputer(strategy="constant",fill_value=0.0)
            final_data_prepared = imputer.fit_transform(nonhomologous_data[numeric_columns])
            final_data_prepared = pd.DataFrame(final_data_prepared, columns=numeric_columns)
            final_data_prepared = pd.merge(vep_scores,final_data_prepared,right_index=True,left_index=True)
            final_data_prepared.dropna(subset=[target_column],inplace=True)
            training_data, testing_data = train_test_split(final_data_prepared.copy(), test_size=0.2,
                                                           random_state=idx + 1)
            y_train, X_train = training_data.pop(target_column), training_data
            # TODO: Check with rounding the dependent variable
            y_train = y_train.round(decimals=2)
            y_test, X_test = testing_data.pop(target_column), testing_data
            # save_testing_genes(args, f"{vep}.{idx + 1}", combined_data, X_test)
            y_test = y_test.round(decimals=2)
            X_train_normalized = X_train
            logger.info(best_params)
            if best_params:
                best_params['max_depth'] = int(round(best_params['max_depth']))
                best_params['n_estimators'] = int(round(best_params['n_estimators']))
                best_params["min_samples_split"] = int(round(best_params["min_samples_split"]))
                model = RandomForestRegressor(**best_params)
            else:
                model = RandomForestRegressor(**default_params)
            model.fit(X_train_normalized, y_train)
            y_pred = model.predict(X_test)
            r_test, _ = st.spearmanr(y_test, y_pred)
            r2_test = m.r2_score(y_test, y_pred)
            r, r2, ev, rmse = get_metrics(model, final_data_prepared, vep)
            if args.verbose:
                logger.info(f"Perfs: spearman: {r_test} , R2:{r2} , EV: {ev} , R2_test:{r2_test}")
            performances.append((vep, rmse, r2, ev, r, r_test, r2_test))
        except Exception as e:
            logger.info(f"Iteration: {idx + 1}")
            raise e
    logger.info(f"Done everything for VEP: {vep}, Saving Performances")
    perf = pd.DataFrame(performances, columns=['vep', 'rmse', 'r2', 'ev', 'all_corr', 'test_corr', 'test_r2'])
    out_perfs_file = os.path.join(args.output,"perfs", f"{vep}.perf.csv")
    perf.to_csv(out_perfs_file, index=False)
    logger.info(f"Started final Training of the VEP: {vep} on all data.....")
    #all_cols = [target_column] + numeric_columns
    imputer = SimpleImputer(strategy="constant",fill_value=0.0)
    vep_scores = all_data[target_column]
    final_data_prepared = imputer.fit_transform(all_data[numeric_columns])
    final_data_prepared = pd.DataFrame(final_data_prepared, columns=numeric_columns)
    final_data_prepared = pd.merge(vep_scores,final_data_prepared,right_index=True,left_index=True)
    final_data_prepared.dropna(subset=[target_column],inplace=True)
    train_the_final_model(args, vep, final_data_prepared, multiple_dir)
    logger.info(f"All Done")


def main():
    p = ArgumentParser(
        description="This program will train individual models and generate performance metrics for the regression "
                    "analysis")
    p.add_argument("-a", "--aucs", help="The Genes with calculated AUCs across all veps.", required=True)
    p.add_argument("-g", "--genes", help="The file with genes with features to load", required=True)
    p.add_argument("-t", "--optimal", help="The optimal Hyperparameters directory", required=True)
    p.add_argument("-l", "--nonhom", help="Non-Homologous Set to use", required=True)
    p.add_argument("-z", "--vep", help="VEP to use", required=True)
    p.add_argument("-B","--verbose",type=bool,default=False,help="Whether to print verbose messages")
    p.add_argument("-o", "--output", help="The output directory where to save the output", required=True)
    if len(sys.argv) <= 1:
        p.print_help()
        return
    args = p.parse_args()
    process(args)


if __name__ == '__main__':
    main()

