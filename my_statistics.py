import numpy as np
import pandas as pd
from scipy import stats
import scikit_posthocs as sp

# Function that returns a summary of Spearman R correlation for selected variable with all other numeric variables
def spearman_assess(data_frame, correlate_with):
    table = pd.DataFrame()
    names = []
    strength = []
    correlation = []
    pvalue = []

    # Calculating Spearman R for all numeric variables
    for column in data_frame.columns:
        if data_frame[column].dtype.kind in 'if':
            if column != correlate_with:
                test = stats.spearmanr(data_frame[correlate_with], data_frame[column], nan_policy="omit")

                # Preparing series for a summary table
                names.append(column)
                correlation.append(test[0])
                pvalue.append(test[1])
                if test[0] >= 0.89 or test[0] <= -0.89:
                    strength.append("Very Strong")
                elif test[0] >= 0.68 or test[0] <= -0.68:
                    strength.append("Strong")
                elif test[0] >= 0.38 or test[0] <= -0.38:
                    strength.append("Moderate")
                elif test[0] >= 0.1 or test[0] <= -0.1:
                    strength.append("Weak")
                else:
                    strength.append("None")

    # Creating a summary table
    table["Feature"] = names
    table["Spearman_R"] = correlation
    table["P_Value"] = pvalue
    table["Strength"] = strength

    return table.reindex(table["Spearman_R"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# Helper functions that convert values to numbers as a preparation for Kruskal-Wallis H test
def write_names(data_frame, column):
    names = {}
    numbers = {}
    cat_nr = 1
    for name in data_frame[column].unique():
        if name != np.nan:
            numbers[name] = cat_nr
            names[cat_nr] = name
            cat_nr += 1
    return numbers, names


def swap(data_frame, column, to_numbers=True):
    to_num, to_cat = write_names(data_frame, column)
    if to_numbers:
        return data_frame[column].apply(lambda x: to_num[x])
    else:
        return data_frame[column].apply(lambda x: to_cat[x])


# Function that computes Kruskal-Wallis H for selected pair of columns
def kruskal_one(data_frame, column, correlate_with):
    to_num, to_cat = write_names(data_frame, column)
    table = data_frame.copy()
    table[column] = swap(table, column, to_num)
    return stats.kruskal(table[correlate_with], table[column], nan_policy="omit")


# Function that computes Kruskal-Wallis H for selected column with all categorical variables
def kruskal_all(data_frame, correlate_with):
    table = pd.DataFrame()
    names = []
    hvalue = []
    pvalue = []

    # Calculating Spearman R for all numeric variables
    for name in data_frame.columns:
        if data_frame[name].dtype.kind not in 'if':
            if name != correlate_with:
                test = kruskal_one(data_frame, name, correlate_with)
                # Preparing series for a summary table
                names.append(name)
                hvalue.append(test[0])
                pvalue.append(test[1])
    # Creating a summary table
    table["Feature"] = names
    table["Kruskal-Wallis_H"] = hvalue
    table["P_Value"] = pvalue

    return table




