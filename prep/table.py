"""The staff table of the repository: its missing values and whether the columns are related at all."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

NUMERIC = ["Age", "Salary_NPR", "Experience_Years", "Rating"]
CATEGORICAL = ["District", "Department"]


def load_table(path):
    """The table with Join_Date parsed; the other columns as stored."""
    table = pd.read_csv(path)
    missing = set(NUMERIC + CATEGORICAL + ["Join_Date"]) - set(table.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    table["Join_Date"] = pd.to_datetime(table["Join_Date"], format="%m/%d/%Y")
    return table


def missing_summary(table):
    """Missing values per column, and the number of rows by how many values they lack."""
    columns = NUMERIC + CATEGORICAL
    lacking = table[columns].isna()
    per_row = lacking.sum(axis=1).value_counts().sort_index()
    return {
        "rows": int(len(table)),
        "per_column": {name: int(lacking[name].sum()) for name in columns},
        "share_of_cells": float(lacking.to_numpy().mean()),
        "rows_by_missing_values": {int(k): int(v) for k, v in per_row.items()},
        "complete_rows": int((~lacking.any(axis=1)).sum()),
    }


def holm(p_values):
    """Holm's step-down adjustment of a set of p-values, in the order given."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    stepped = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(stepped, 1.0)
    return adjusted


def chi_square_permutation(labels, groups, rng, draws=2000):
    """p-value of the chi-square statistic of a two-way table, by permuting one of the two columns.

    Used where expected counts are too small for the chi-square distribution.
    """
    rows = pd.factorize(labels)[0]
    cols = pd.factorize(groups)[0]
    shape = (rows.max() + 1, cols.max() + 1)

    def statistic(columns):
        observed = np.zeros(shape)
        np.add.at(observed, (rows, columns), 1)
        expected = observed.sum(axis=1, keepdims=True) * observed.sum(axis=0, keepdims=True) / observed.sum()
        return ((observed - expected) ** 2 / expected).sum()

    actual = statistic(cols)
    exceed = sum(statistic(rng.permutation(cols)) >= actual - 1e-12 for _ in range(draws))
    return float((exceed + 1) / (draws + 1))


def _relation_p(first, second, rng):
    """p-value for a relation between two columns; the test depends on their kinds."""
    known = first.notna() & second.notna()
    first, second = first[known], second[known]
    numeric = [pd.api.types.is_numeric_dtype(column) and column.nunique() > 2 for column in (first, second)]
    if all(numeric):
        return "Spearman", float(stats.spearmanr(first, second).pvalue)
    if any(numeric):
        value, group = (first, second) if numeric[0] else (second, first)
        parts = [part.to_numpy() for _, part in value.groupby(group.to_numpy())]
        if len(parts) == 2:
            return "Mann-Whitney", float(stats.mannwhitneyu(*parts).pvalue)
        return "Kruskal-Wallis", float(stats.kruskal(*parts).pvalue)
    return "chi-square, permuted", chi_square_permutation(first.to_numpy(), second.to_numpy(), rng)


def relation_tests(table, rng):
    """A test of every pair of columns for a relation between their values, with Holm-adjusted p-values."""
    columns = NUMERIC + CATEGORICAL
    rows = []
    for i, first in enumerate(columns):
        for second in columns[i + 1:]:
            test, p = _relation_p(table[first], table[second], rng)
            rows.append({"first": first, "second": second, "test": test, "p": p})
    result = pd.DataFrame(rows)
    result["p_holm"] = holm(result["p"])
    return result


def missingness_tests(table, rng):
    """A test of whether a value is more often missing in one column depending on the value in another.

    One test per incomplete column and other column, with Holm-adjusted p-values.
    """
    columns = NUMERIC + CATEGORICAL
    rows = []
    for incomplete in columns:
        lacking = table[incomplete].isna()
        if not lacking.any():
            continue
        for other in columns:
            if other != incomplete:
                test, p = _relation_p(lacking.astype(int), table[other], rng)
                rows.append({"missing_in": incomplete, "other": other, "test": test, "p": p})
    result = pd.DataFrame(rows)
    result["p_holm"] = holm(result["p"])
    return result


def consistency(table):
    """Facts that a record of real staff would not show."""
    both = table.dropna(subset=["Age", "Experience_Years"])
    days = table["Join_Date"].diff().dt.days.dropna()
    spearman = table[NUMERIC].corr(method="spearman").to_numpy()
    return {
        "rows_with_age_and_experience": int(len(both)),
        "experience_above_age": int((both["Experience_Years"] > both["Age"]).sum()),
        "experience_above_age_minus_15": int((both["Experience_Years"] > both["Age"] - 15).sum()),
        "join_dates_are_consecutive_days": bool((days == 1).all()),
        "first_join_date": f"{table['Join_Date'].min():%Y-%m-%d}", "last_join_date": f"{table['Join_Date'].max():%Y-%m-%d}",
        "largest_rank_correlation_between_numeric_columns": float(np.abs(spearman[np.triu_indices(len(NUMERIC), k=1)]).max()),
        # a uniform distribution has an excess kurtosis of -1.2, a normal one of 0
        "excess_kurtosis": {name: float(stats.kurtosis(table[name].dropna())) for name in NUMERIC},
    }
