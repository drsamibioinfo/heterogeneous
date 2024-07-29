import os, sys
import logging
from argparse import ArgumentParser
import os, json, joblib as jl
import glob
import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from pandas.api.types import is_numeric_dtype
from sklearn import metrics as m
from sklearn.model_selection import train_test_split
import warnings
from bayes_opt import BayesianOptimization, JSONLogger
from bayes_opt.event import Events
import json
import optuna


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

logger = logging.getLogger("Hyperparameter Optimization")


def stratified_kfold_r2_score(estimator, X, y, n_fold):
    kfold = KFold(n_splits=n_fold, shuffle=True, random_state=1)
    r2_list = []

    for train_index, test_index in kfold.split(X, y):
        x_train_fold, x_test_fold = X.loc[train_index], X.loc[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        estimator.fit(x_train_fold, y_train_fold)
        r2 = estimator.score(x_test_fold, y_test_fold)
        r2_list.append(r2)

    return np.nanmean(r2_list)


def kfold_with_abs_error(estimator, X, y, n_fold):
    kfold = KFold(n_splits=n_fold, shuffle=True, random_state=1)
    abs_errors = []

    for train_index, test_index in kfold.split(X, y):
        x_train_fold, x_test_fold = X.loc[train_index], X.loc[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        estimator.fit(x_train_fold, y_train_fold)
        preds = estimator.predict(x_test_fold)
        errors = abs(preds - y_test_fold)
        abs_errors.append(np.mean(errors))
    return np.mean(abs_errors)


def stratified_MSE(estimator, X, y, n_fold):
    from sklearn.model_selection import cross_val_score
    score = cross_val_score(estimator, X, y, n_jobs=-1, cv=n_fold,scoring="neg_mean_squared_error")
    return np.mean(-score)


def stratified_kfold_score(estimator, X, y, n_fold):
    kfold = KFold(n_splits=n_fold, shuffle=True, random_state=1)
    mse_list = []

    for train_index, test_index in kfold.split(X, y):
        x_train_fold, x_test_fold = X.loc[train_index], X.loc[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        estimator.fit(x_train_fold, y_train_fold)
        training_score = estimator.score(x_train_fold,y_train_fold)
        testing_score = estimator.score(x_test_fold,y_test_fold)
        current_score = (training_score - testing_score) ** 2
        mse_list.append(current_score)
    mse = np.nanmean(mse_list)
    return np.sqrt(mse)


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


def process(args):
    genes = pd.read_csv(args.genes)
    aucs = pd.read_csv(args.aucs)
    numeric_columns = [x for x in genes.columns if (
            is_numeric_dtype(genes[x].dtype) and x not in ["start_position", "end_position",
                                                           "entrezgene_id"])]
    df = pd.merge(aucs[['gene', args.vep]], genes[['gene'] + numeric_columns], how='left', on='gene')
    df.dropna(subset=[args.vep], inplace=True)
    df = rename_redundant_features(df,args.vep)
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    logger.info(f"Working on: {args.vep}")
    df.drop('gene', inplace=True, axis=1)
    df_imputed = imputer.fit_transform(df)
    df_imputed = pd.DataFrame(df_imputed, columns=df.columns)
    Y, X = df_imputed.pop(args.vep), df_imputed

    # define the objective function
    def objective(trial: optuna.trial.Trial):
        n_estimators = trial.suggest_int('n_estimators', 100, 4000)
        max_depth = trial.suggest_int('max_depth', 1, 300)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 50)
        criterion = trial.suggest_categorical("criterion", ["squared_error", "friedman_mse", "poisson"])
        max_features = trial.suggest_categorical("max_features", [None, "sqrt", "log2"])
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      criterion=criterion,
                                      max_features=max_features,
                                      min_samples_split=min_samples_split)
        #score, _ = stratified_kfold_score(model, X=X.reset_index(), y=Y.reset_index()[args.vep], n_fold=5)
        score = stratified_kfold_score(model, X=X.reset_index(), y=Y.reset_index()[args.vep], n_fold=5)
        return score

    logger.info(f"Performing {args.trials} Optimization Trials.")
    sampler = optuna.samplers.QMCSampler()
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=args.trials)
    logger.info(f"Done Performing Optimization for {args.vep}")
    vep_file_name = os.path.join(args.output, f"{args.vep}.params.txt")
    with open(vep_file_name, "w") as writer:
        best_params_string = json.dumps(study.best_params)
        writer.write(best_params_string)
    logger.info(f"Done Writing the best hyperparameters file to: {vep_file_name}")


def main():
    p = ArgumentParser(
        description="This program will perform hyperparameter optimization for a single predictor passed to the value "
                    "named 'vep'")
    p.add_argument("-a", "--aucs", help="The Genes with calculated AUCs across all veps.", required=True)
    p.add_argument("-g", "--genes", help="The file with genes with features to load", required=True)
    p.add_argument("-e", "--vep", help="The Predictor name to perform hyperparameter optimization for.", required=True)
    p.add_argument("-t", "--trials", help="The number of Trials to optimize. Defaults to 300 Trials", default=300,
                   type=int)
    p.add_argument("-o", "--output", help="The output directory where to save the output", required=True)
    if len(sys.argv) <= 1:
        p.print_help()
        return
    args = p.parse_args()
    process(args)


if __name__ == '__main__':
    main()

