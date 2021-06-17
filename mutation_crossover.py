"""
Abdo Alnabulsi, University of Aberdeen/ AiBIOLOGICS 2021. see separate LICENSE.
This is an experimental software and should be used with caution.

This Python script is associated with the following project ""Identification of a prognostic signature
in colorectal cancer using combinatorial algorithms driven analysis".Please refer to the article and the
supplementary material for a detailed description of the study.

This script implements an evolutionary algorithm.
The main aim of combinatorial code is to iterate over random combinations of biomarkers, testing their association with
survival, selecting parents based on roulette/tournament, first child is union of parents, other children are created
through random mutation and crossover of first child, and finally picking up children with the best fitness
(association mainly using Cox reg). Here, the hazard ratio was the
main measure of performance. However, others available via can be used (e.g.coefficients, AIC, model accuracy etc..).
"""

import os
import logging

import numpy as np
import pandas as pd
import random
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


def roulette_wheel_selection(population):
    chosen = []
    fitness = population.values()
    total_fit = float(sum(fitness))
    relative_fitness = [f / total_fit for f in fitness]
    probabilities = [sum(relative_fitness[:i + 1])
                     for i in range(len(relative_fitness))]
    for n in range(2):
        r = random.random()
        for (i, individual) in enumerate(population):
            if r <= probabilities[i]:
                chosen.append(list(individual))
                break
    return chosen


def tournament(population):
    couple = []
    for i in range(2):
        candidates = random.sample(population.keys(), int(len(population.keys()) * 0.5))
        parent = max(candidates, key=population.get)
        del population[parent]
        couple.append(list(parent))
    return couple


def mutation(initial_population, first_child, children_num, mut_rate):
    """ function to mutate selected combination by deleting, adding and replacing variables at a mut_rate."""
    mut_vars = list(set(initial_population).difference(set(first_child)))
    mut = round(len(first_child) * mut_rate)
    if mut == 0 or mut_vars == 0:
        return [initial_population[0:round(len(initial_population) * 0.8)], first_child]
    else:
        if mut > len(mut_vars):
            mut = len(mut_vars)
        mut_children = []
        for x in range(children_num):
            mut_children.append(first_child[:])
        slic = children_num // 3
        # deletion children
        for child in mut_children[0:slic]:
            for _ in range(mut):
                child.pop(random.randint(0, len(child) - 1))
        # insertion
        for child in mut_children[slic:2 * slic]:
            child.extend(random.sample(mut_vars, k=mut))
        # replacement
        for child in mut_children[2 * slic:3 * slic]:
            for x in range(mut):
                child[x] = mut_vars[x]
        return mut_children


def cross_over(parent1, parent2, crossover_num):
    """ implement crossing over between parents at a given cross over rate and return child."""
    crossed = list(set(parent2).difference(set(parent1)))
    if crossover_num == 0 or crossed == 0 or set(parent2).issubset(set(parent1)):
        return parent1
    else:
        cross_child = parent1[:]
        if crossover_num > len(crossed):
            crossover_num = len(crossed)
        for x in range(crossover_num):
            cross_child.pop(random.randint(0, len(cross_child) - 1))
            cross_child.append(crossed[x])
        return cross_child


def generation(df, initial_population, variables, composite_compute, combinations_num, parent_selection_method,
               children_num, mut_rate, crossover_num, total_model, model, drop_missing_rate, interpolation_method,
               train_test, multi, test):
    """ generate random combinations, select best as parents, generate children and perform mutations and crossovers
     and select the best combination to to the next generation."""
    logging.info("[generation] finding best child in each generation")
    gen = []
    for _ in range(combinations_num):
        # Randomly create a binomial random array of zeros and ones: ones variables being selected and no for zeros.
        combination_selection = np.random.binomial(1, 0.9, len(variables))
        var = []
        for x in range(len(variables)):
            if combination_selection[x] == 1:
                var.append(variables[x])
        gen.append(var)
    parents = {}
    for combination in gen:
        if len(combination) > 2:
            result = test_combination(df, combination, total_model, model, composite_compute,  drop_missing_rate,
                                      interpolation_method, train_test, multi, test)
            parents[tuple(combination)] = result[1]
            gen.remove(combination)

    if parent_selection_method == 'tournament':
        couple = tournament(parents)
    else:
        couple = roulette_wheel_selection(parents)

    first_child = list(set(couple[0]).intersection(set(couple[1])))
    best_solution = test_combination(df, first_child, total_model, model, composite_compute, drop_missing_rate,
                                     interpolation_method, train_test, multi, test)

    crossover = cross_over(couple[0], couple[1], crossover_num)
    crossover_outcome = test_combination(df, crossover, total_model, model, composite_compute, drop_missing_rate,
                                         interpolation_method, train_test, multi, test)
    if crossover_outcome[1] > best_solution[1]:
        best_solution = crossover_outcome

    mut_children = mutation(initial_population, first_child, children_num, mut_rate)
    for child in mut_children:
        if len(child) < 2:
            continue
        outcome = test_combination(df, child, total_model, model, composite_compute, drop_missing_rate,
                                   interpolation_method, train_test, multi, test)
        if outcome[1] > best_solution[1]:
            best_solution = outcome
    return best_solution


def optimal_generation(df, initial_population, variables, combinations_num, children_num, composite_compute='cluster',
                       parent_selection_method='roulette', total_model=True, model=None, crossover_num=1,
                       mut_rate=0.2, train_test=False, drop_missing_rate=1.0, interpolation_method='linear',
                       iteration=1000, multi=True, test='cox', sac_rate=1.0):
    """ random combinations, select best combinations, intersection, crossovers,mutations,
    find fittest combination which is taken forward for next generation.
    Parameters
    ----------
    df: Dataframe
    initial_population:list
                    initial population of variables/markers.
    variables: list
            list of markers / variables in a generation.
    combinations_num, children_num:int
    parent_selection_method: str, default roulette
    composite_compute:str, default cluster
    total_model:bol, default True
                the fitness score is based on of the score of model (e.g.c_index) compared to the score of only the
                variables/combination (e.g.coef).
    model:Cox model classifier, default CoxPHSurvivalAnalysis()
    train_test: bol,default True
    mut_rate:float, default 0.2
              determines the proportion of a combination to mutate.
    crossover_num: int, default 1
                determines how many elements to crossover.
    drop_missing_rate: float, default 1.0
                    determines the proportion of non missing values when dropping rows with missing values.
                    default 1.0 means all rows has to have 100% non missing values and will be dropped otherwise.
    interpolation_method: str, default 'linear'
                        based on pandas interpolate method, for details of interpolation methods see:
                            https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html.
    iteration:int, default 1000
    sac_rate:float, default 1.0
            rate of score willing to sacrifice when assessing combinations in different generations, range:0.7-0.99.
    multi: bol, default True
        determines whether to evaluate a multivariate model or a uni-variate model which only include
        the concerned variable/combination. Important to establish independence of existing clinical parameters.
    test: str, default cox.
    """
    logging.info("[optimal_generation] Finding fittest child")

    if not model:
        model = CoxPHSurvivalAnalysis()
    fittest_child = generation(df, initial_population, variables, composite_compute, combinations_num,
                               parent_selection_method, children_num, mut_rate, crossover_num, total_model, model,
                               drop_missing_rate, interpolation_method, train_test, multi, test)
    logging.info('[iteration 0] Best solution: %s', fittest_child)
    for i in range(iteration):
        current_child = generation(df, initial_population, fittest_child[0], composite_compute, combinations_num,
                                   parent_selection_method, children_num, mut_rate, crossover_num, total_model, model,
                                   drop_missing_rate, interpolation_method, train_test, multi, test)
        if current_child[1] / len(current_child[0]) >= (fittest_child[1] / len(current_child[0])) * sac_rate:
            fittest_child = current_child
        logging.info('[iteration %s] Best solution: %s', i, fittest_child)

    return fittest_child


if __name__ == '__main__':
    data = pd.read_excel("data/<>.xlsx")

    """" list of markers/variables. With large number of features (e.g.genomic), it is advisable to apply
        dimensionality reduction and sub_setting of variables to minimise computational  complexity and false
         discoveries: sig_markers and markers_subset"""
    all_markers = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                   'v', 'w', 'x', 'y', 'z', 'a1', 'b1', 'c1', '.....', 'd1', 'pn', 'qn', 'rn']

    # markers with significant univariate association with survival
    sig_markers = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                   'v', 'w', 'x', 'y', 'z', 'a1', 'b1', 'c1']

    # markers subset.
    markers_subset = ['b', 'c', 'd', 'e', 'u', 'x', 'a1', 'b1', 'c1']

    optimal_generation(data, markers_subset, markers_subset, 30, 12, total_model=False, train_test=False,
                       sac_rate=0.9)
