"""
Abdo Alnabulsi, University of Aberdeen/ AiBIOLOGICS 2021. see separate LICENSE.
This is an experimental software and should be used with caution.

This Python script is associated with the following project ""Identification of a prognostic signature
in colorectal cancer using combinatorial algorithms driven analysis".Please refer to the article and the
supplementary material for a detailed description of the study.

This script is a combinatorial algo. Combinations are assessed by creating a composite variable out of the biomarkers
in the combination using linear function or a cluster method.
The main aim of combinatorial code is to iterate over combinations of biomarkers, testing their association with
survival and picking up the combination with the best association mainly using Cox reg. Here, the hazard ratio was the
main measure of performance. However, others available via can be used (e.g.coefficients, AIC, model accuracy etc..).

"""

import os
import itertools
import logging

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxPHSurvivalAnalysis
from lifelines.statistics import pairwise_logrank_test
from sklearn.cluster import AgglomerativeClustering

os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '4'

logging.basicConfig(level=logging.INFO)


def impute_missing(df, variables, drop_missing_rate, interpolation_method, multi):
    """ Drop rows with missing values at drop_missing_rate and interpolate the remaining missing."""

    if multi:
        var_df = pd.concat([df[["Survival", "Status", "agerecoded", "Tstagen", 'Nstagen', 'EMVI']], df[variables]],
                           axis=1)
    else:
        var_df = pd.concat([df[["Survival", "Status"]], df[variables]], axis=1)
    var_df.dropna(thresh=round(len(var_df.columns) * drop_missing_rate), inplace=True)
    if drop_missing_rate < 1.0:
        var_df.interpolate(method=interpolation_method, inplace=True)
        var_df.dropna(inplace=True)
    var_df.reset_index(drop=True, inplace=True)
    return var_df


def get_x_y(df, variables, multi):
    """ x and y out of dataframe using combination of features and survival outputs."""
    df['new_col'] = df.loc[:, ('Status', 'Survival')].apply(tuple, axis=1)
    y = df[['new_col']].to_numpy(dtype=[('Status', '?'), ('Survival', 'int')])
    y = np.squeeze(y)
    if multi:
        x = pd.concat([df[["agerecoded", "Tstagen", "Nstagen", "EMVI"]], df[variables]], axis=1)
        return x, y
    else:
        return df[variables], y


def evaluate_model_accuracy(df, features, model, train_test, multi):
    """ Evaluate accuracy score of models using classifier models taking x, y."""
    x, y = get_x_y(df, features, multi)
    if train_test:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y["Status"], random_state=0)
    else:
        x_train, x_test, y_train, y_test = x, x, y, y
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)


def compute_fitness(df, variables, composite_compute, multi, test):
    """ Variable combination is assessed using a composite variable made of combination of variables (linear function),
    or cluster analysis of variables. Other methods of combining variables can be implemented."""

    if composite_compute == 'linear':
        cph = CoxPHFitter()
        cph.fit(df, 'Survival', 'Status')
        coefficients = list(round(cph.params_[-len(variables):], 2))
        comb = df[variables].mul(coefficients)
        composite = comb.sum(axis=1, skipna=False)
        composite_categorised = pd.qcut(composite, q=5, duplicates='drop', labels=['0', '1', '2', '3', '4'])
    else:
        x = df[variables]
        model = AgglomerativeClustering(n_clusters=5, linkage='ward')
        model.fit(x)
        composite_categorised = model.labels_

    if multi:
        comb_df = df[["Survival", "Status", "agerecoded", "Tstagen", "Nstagen", "EMVI"]].assign(
            comb1=composite_categorised)
        cph1 = CoxPHFitter().fit(comb_df, 'Survival', 'Status', formula="C(comb1)+ agerecoded+EMVI+Nstagen+Tstagen")
        return float(cph1.hazard_ratios_[:-4].values.sum())
    else:
        comb_df = df[["Survival", "Status"]].assign(comb1=composite_categorised)
        if test == 'logrank':
            result = pairwise_logrank_test(comb_df['Survival'], comb_df['comb1'], comb_df['Status'])
            return float(result.summary['-log2(p)'].values[0:4].sum())
        else:
            cph1 = CoxPHFitter().fit(comb_df, 'Survival', 'Status', formula='comb1')
            return float(cph1.hazard_ratios_.values.sum())


def test_combination(df, variables, total_model, model, composite_compute, drop_missing_rate, interpolation_method,
                     train_test, multi, test):

    df = impute_missing(df, variables, drop_missing_rate, interpolation_method, multi)
    if total_model:
        fitness = evaluate_model_accuracy(df, variables, model, train_test, multi)
    else:
        fitness = compute_fitness(df, variables, composite_compute, multi, test)
    return variables, fitness


def compare_comb_stepwise_global(df, variables, step, total_model, model, composite_compute,
                                 drop_missing_rate, interpolation_method, train_test, multi, test):
    logging.info("[stepwise_global] Compare combinations")
    best_score = 0
    best_solution = None
    for x in range(len(variables) - 1, len(variables) - step, -1):
        logging.info('Iteration: %s', x)
        population = itertools.combinations(variables, x)

        for combination in population:
            current_solution = test_combination(df, list(combination), total_model, model, composite_compute,
                                                drop_missing_rate, interpolation_method, train_test, multi, test)
            if current_solution[1] > best_score:
                best_score = current_solution[1]
                best_solution = current_solution
    return best_solution


def optimal_comb_stepwise_global(df, variables, step=3, total_model=True, model=None, composite_compute='cluster',
                                 drop_missing_rate=1.0, interpolation_method='linear', generation_num=100,
                                 train_test=False, sac_rate=1.0, multi=True, test='cox'):
    """ generate nCr = n! / r! * (n - r)! with r = len(variables)-s and find best combination to take forward to
        next generation. Repeat until convergence.
    Parameters
    ----------
    df: Dataframe
    variables: list
            list of markers / variables in a generation.
    step:int, default 4
        defines the range of markers number in a combination. If s=len(variables)-1 then all possible
            combinations are generated.
    total_model:bol, default True
                the fitness score is based on of the score of model (e.g.c_index) versus the score of only
                the variables/combination (e.g.coef).
    composite_compute:str default cluster
    model:linear models, default CoxPHSurvivalAnalysis()
    train_test: bol,default True
    drop_missing_rate: float, default 1.0
                    determines the proportion of non missing values when dropping rows with missing values.
                    default 1.0 means all rows has to have 100% non missing values and will be dropped otherwise.
    interpolation_method: str, default 'linear'
            This is based on pandas interpolate method, for details of interpolation methods https://pandas.pydata.org/
            docs/reference/api/pandas.DataFrame.interpolate.html.
    generation_num:int, default 100
    sac_rate:float, default 1.0
            rate of score willing to sacrifice when assessing combinations in different generations, range:0.7-0.99.
    multi: bol, default True
        determines whether to evaluate a multivariate model or a uni-variate model which only include
        the concerned variable/combination. Important to establish independence of existing clinical parameters.
    test: str, default cox.
    """
    logging.info("[optimal_comb_stepwise_global] Finding best solution")
    if not model:
        model = CoxPHSurvivalAnalysis()

    best_solution = compare_comb_stepwise_global(df, variables, step, total_model, model, composite_compute,
                                                 drop_missing_rate, interpolation_method, train_test, multi, test)
    logging.info('[iteration 0] Best solution: %s', best_solution)
    for i in range(generation_num):
        current_solution = compare_comb_stepwise_global(df, best_solution[0], step, total_model, model,
                                                        composite_compute, drop_missing_rate, interpolation_method,
                                                        train_test, multi, test)

        if current_solution[1] / len(current_solution[0]) ** 2 >= (best_solution[1] /
                                                                   len(best_solution[0]) ** 2) * sac_rate:
            best_solution = current_solution
            logging.info('[iteration %s] Best solution: %s', i + 1, best_solution)
            if len(best_solution[0]) - step <= 1:
                break


def compare_comb_gradient(df, variables, total_model, model, composite_compute, drop_missing_rate, interpolation_method,
                          train_test, multi, test):
    logging.info("[gradient] Compare combinations")
    population = itertools.combinations(variables, len(variables) - 1)
    best_fit = 0
    best_solution = None
    for combination in population:

        current_comb = test_combination(df, list(combination), total_model, model, composite_compute, drop_missing_rate,
                                        interpolation_method, train_test, multi, test)
        if current_comb[1] > best_fit:
            best_fit = current_comb[1]
            best_solution = current_comb
    return best_solution


def optimal_comb_gradient(df, variables, total_model=True, model=None, composite_compute='cluster',
                          drop_missing_rate=1.0, interpolation_method='linear', generation_num=100, train_test=False,
                          sac_rate=1.0, multi=True, test='cox'):
    """ generate nCr = n! / r! * (n - r)! with r = len(variables)-1 and find best combination to take forward to
    next generation. Repeat until convergence.
    Parameters
    ----------
    df: Dataframe
    variables: list
            list of markers / variables in a generation.
    total_model:bol, default True
                the fitness score is based on of the score of model (e.g.c_index) compared to the score of only the
                variables/combination (e.g.coef).
    composite_compute: str, default cluster
    model:Linear models: CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis , default CoxPHSurvivalAnalysis
    train_test: bol,default True
    drop_missing_rate: float, default 1.0
                    determines the proportion of non missing values when dropping rows with missing values.
                    default 1.0 means all rows has to have 100% non missing values and will be dropped otherwise.
    interpolation_method: str, default 'linear'
                        based on pandas interpolate method, for details of interpolation methods see:
                        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html.
    generation_num:int, default 100
    sac_rate:float, default 1.0
            rate of score willing to sacrifice when assessing combinations in different generations, range:0.7-0.99.
    multi: bol, default True
        determines whether to evaluate a multivariate model or a uni-variate model which only include
        the concerned variable/combination. Important to establish independence of existing clinical parameters.
    test: str, default cox.
    """
    logging.info("[optimal_comb_gradient] Finding best solution")
    if not model:
        model = CoxPHSurvivalAnalysis()
    best_solution = compare_comb_gradient(df, variables, total_model, model, composite_compute, drop_missing_rate,
                                          interpolation_method, train_test, multi, test)
    logging.info('[iteration 0] Best solution: %s', best_solution)
    for i in range(generation_num):
        current_solution = compare_comb_gradient(df, list(best_solution[0]), total_model, model, composite_compute,
                                                 drop_missing_rate, interpolation_method, train_test, multi, test)
        if current_solution[1] / len(current_solution) ** 2 >= (best_solution[1] / len(best_solution) ** 2) * sac_rate:
            best_solution = current_solution
            logging.info('[iteration %s] Best solution: %s', i, best_solution)
            if len(best_solution[0]) == 2:
                break


if __name__ == '__main__':
    data = pd.read_excel("data/<>.xlsx")

    """" list of markers/variables. With large number of features (e.g.genomic), it is advisable to apply
        dimensionality reduction and sub_setting of variables to minimise computational  complexity and false
         discoveries: sig_markers and markers_subset."""
    all_markers = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                   'v', 'w', 'x', 'y', 'z', 'a1', 'b1', 'c1','d1', 'pn', 'qn', 'rn']

    # markers with significant univariate association with survival
    sig_markers = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's','t', 'u',
                   'v', 'w', 'x', 'y', 'z', 'a1', 'b1', 'c1']

    # markers subset.
    markers_subset = ['b', 'c', 'd', 'e','u', 'x', 'a1', 'b1', 'c1']

    optimal_comb_gradient(data, markers_subset, total_model=True, sac_rate=0.8)
