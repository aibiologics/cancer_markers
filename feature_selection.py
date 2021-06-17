"""
Abdo Alnabulsi, University of Aberdeen/ AiBIOLOGICS 2021. see separate LICENSE.
This is an experimental software and should be used with caution.

This Python script is associated with the following project ""Identification of a prognostic signature
in colorectal cancer using combinatorial algorithms driven analysis".Please refer to the article and the
supplementary material for a detailed description of the study.

This script is an adaptation of feature selection based https://scikit-learn.org/stable/auto_examples/inspection/
plot_permutation_importance_multicollinear.html.
The main aim of this code is to iterate over biomarkers in each cluster (correlation based cluster) removing one
biomarker from each cluster, then testing the remaining biomarkers in terms of association with survival.
Here, the hazard ratio was the main measure of performance. However, others available via can be used
(e.g.coefficients, AIC, model accuracy etc..).

"""


import os
import logging
import random

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxPHSurvivalAnalysis
from lifelines import CoxPHFitter
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sksurv.preprocessing import OneHotEncoder
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.cluster import hierarchy

os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '4'

logging.basicConfig(level=logging.INFO)


def impute_missing(df, features, drop_missing_rate, interpolation_method):
    """drop  rows with missing values using threshold of drop_missing_rate, and interpolate the remaining missing."""

    var_df = pd.concat([df[["Survival", "Status", "agerecoded", "Tstagen", "Nstagen", "EMVI"]],
                        df[features]], axis=1)
    var_df.dropna(thresh=round(len(var_df.columns) * drop_missing_rate), inplace=True)
    if drop_missing_rate < 1.0:
        var_df.interpolate(method=interpolation_method, inplace=True)
        var_df.dropna(inplace=True)
    var_df.reset_index(drop=True, inplace=True)
    return var_df


def get_x_y(df, features):
    """ x and y out of dataframe using combination of features and survival outputs."""
    df['new_col'] = df.loc[:, ('Status', 'Survival')].apply(tuple, axis=1)
    y = df[['new_col']].to_numpy(dtype=[('Status', '?'), ('Survival', 'int')])
    y = np.squeeze(y)
    x = df[features]
    return x, y


def evaluate_model_accuracy(df, features, model, drop_missing_rate, interpolation_method, train_test):
    """ evaluate the accuracy score (e.g.c_index) of model made of combination of features."""
    df = impute_missing(df, features, drop_missing_rate, interpolation_method)
    x, y = get_x_y(df, features)
    if train_test:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y["Status"], random_state=0)
    else:
        x_train, x_test, y_train, y_test = x, x, y, y
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)


def plt_corr_cluster(df, features, drop_missing_rate, interpolation_method):
    """ plot clusters of correlated features."""
    df = impute_missing(df, features, drop_missing_rate, interpolation_method)
    x, y = get_x_y(df, features)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    corr = spearmanr(x).correlation
    corr_linkage = hierarchy.ward(corr)
    dendro = hierarchy.dendrogram(corr_linkage, labels=x.columns, ax=ax1, leaf_rotation=90)
    dendro_idx = np.arange(0, len(dendro['ivl']))
    ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
    ax2.set_yticklabels(dendro['ivl'])
    fig.tight_layout()
    plt.show()


def evaluate_features(df, features, model, total_model, drop_missing_rate, interpolation_method, iteration, train_test):
    """ perform hierarchical clustering on the Spearman rank-order correlations of features, then picking up features
    from each cluster with the least effect on the accuracy of the model."""

    logging.info("[evaluate_features] cluster correlated features and select best")

    best_features = None
    df = impute_missing(df, features, drop_missing_rate, interpolation_method)
    x, y = get_x_y(df, features)
    if train_test:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y["Status"], random_state=0)
    else:
        x_train, x_test, y_train, y_test = x, x, y, y

    corr = spearmanr(x).correlation
    corr_linkage = hierarchy.ward(corr)
    cluster_ids = hierarchy.fcluster(corr_linkage, 1, criterion='distance')
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    best_score = 0
    for _ in range(iteration):
        selected_features = [v[random.randint(0, len(v) - 1)] for v in cluster_id_to_feature_ids.values()]
        selected_variables = []
        for i in selected_features:
            selected_variables.append(x.columns[i])
        if total_model:
            x_train_sel = x_train.iloc[:, selected_features]
            x_test_sel = x_test.iloc[:, selected_features]
            model.fit(x_train_sel, y_train)
            current = model.score(x_test_sel, y_test)
            if current > best_score:
                best_score = current
                best_features = selected_variables
            return best_features, best_score
        else:
            cph = CoxPHFitter()
            df.drop('new_col', axis=1, inplace=True)
            cph.fit(df, 'Survival', 'Status')
            coefficients = list(round(cph.params_[-len(selected_features):], 2))
            comb = df[selected_variables].mul(coefficients)
            composite = comb.sum(axis=1, skipna=False)
            composite_categorised = pd.cut(composite, bins=5, labels=['0', '1', '2', '3', '4'])
            comb_df = df[["Survival", "Status"]].assign(comb1=composite_categorised)
            cph1 = CoxPHFitter().fit(comb_df, 'Survival', 'Status')
            current = abs(cph1.hazard_ratios_[0])
            if current > best_score:
                best_score = current
                best_features = selected_variables
            return best_features, best_score


def select_features(df, features, model=None, total_model=True, plot=False, drop_missing_rate=1.0,
                    interpolation_method='linear', generation=30, iteration=1000, train_test=True, sac_rate=1.0):
    """deal with collinear features: removing correlated features and evaluating the score of the model each time
    thereby picking up the least correlated features with the best model score possible.
    Parameters
    ----------
    df: Dataframe
    features: list
            list of markers / variables.
    total_model:bol, default True
                the fitness score is based on of the score of model (e.g.c_index) versus the score of only the
                variables/combination (e.g.coef).
    model:Cox model classifier, default CoxPHSurvivalAnalysis()
    plot: bol, default False
        defines whether to plot clusters.
    train_test: bol,default True
    drop_missing_rate: float, default 1.0
                    determines the proportion of non missing values when dropping rows with missing values.
                    default 1.0 means all rows has to have 100% non missing values and will be dropped otherwise.
    interpolation_method: str, default 'linear'
                        based on pandas interpolate method, for details of interpolation methods see:
                        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html.
    generation, iteration:int, default 1000
    sac_rate:float, default 1.0
             rate of score willing to sacrifice when assessing combinations in different generations, range:0.7-0.99.
    """

    logging.info("[select_features] find optimal number of least correlated features")
    if not model:
        model = CoxPHSurvivalAnalysis()
    initial_score = evaluate_model_accuracy(df, features, model, drop_missing_rate, interpolation_method, train_test)
    logging.info('[initial_score] evaluate model before removing any features: %s', initial_score)

    if plot:
        plt_corr_cluster(df, features, drop_missing_rate, interpolation_method)
    best_features, best_score = evaluate_features(df, features, model, total_model, drop_missing_rate,
                                                  interpolation_method, iteration, train_test)
    logging.info('[iteration 0] Best features: %s', best_features)
    logging.info('[iteration 0] Best score: %s', best_score)
    if plot:
        plt_corr_cluster(df, best_features, drop_missing_rate, interpolation_method)

    for i in range(generation):
        current_features, current_score = evaluate_features(df, best_features, model, total_model, drop_missing_rate,
                                                            interpolation_method, iteration, train_test)
        if current_score / len(current_features) ** 2 > best_score / (len(best_features) ** 2) * sac_rate:
            best_features = current_features
            best_score = current_score
            logging.info('[iteration %s] new best features: %s', i + 1, best_features)
            logging.info('[iteration %s] new best score: %s', i + 1, best_score)
            if plot:
                plt_corr_cluster(df, best_features, drop_missing_rate, interpolation_method)


def fit_score_features(x, y):
    """ which variable is most predictive."""

    n_features = x.shape[1]
    scores = np.empty(n_features)
    m = CoxPHSurvivalAnalysis()
    for j in range(n_features):
        xj = x[:, j:j + 1]
        m.fit(xj, y)
        scores[j] = m.score(xj, y)
    return scores


def best_predictive_features(df, features, drop_missing_rate=1.0, interpolation_method='linear'):
    logging.info("[best_predictive_features] find best predictive features")

    df = impute_missing(df, features, drop_missing_rate, interpolation_method)
    x, y = get_x_y(df, features)

    pipe = Pipeline([('encode', OneHotEncoder()),
                     ('select', SelectKBest(fit_score_features, k=3)),
                     ('model', CoxPHSurvivalAnalysis())])
    param_grid = {'select__k': np.arange(1, x.shape[1] + 1)}
    cv = KFold(n_splits=3, random_state=1, shuffle=True)
    gcv = GridSearchCV(pipe, param_grid, return_train_score=True, cv=cv)
    gcv.fit(x, y)
    pipe.set_params(**gcv.best_params_)
    pipe.fit(x, y)

    encoder, transformer, final_estimator = [s[1] for s in pipe.steps]
    print(pd.Series(final_estimator.coef_, index=encoder.encoded_columns_[transformer.get_support()]))


if __name__ == '__main__':
    data = pd.read_excel("data/<file>.xlsx")
    markers = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
               't', 'u', 'v', 'w', 'x', 'y', 'z', 'a1', 'b1', 'c1']

    select_features(data, markers, total_model=False, sac_rate=0.9)
