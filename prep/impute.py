"""Ways to fill missing values, and experiments that show what each of them does to a table."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier

from .simulate import MECHANISMS, hide, staff_table
from .table import CATEGORICAL, NUMERIC

METHODS = ("mean", "median", "group mean", "nearest neighbours", "regression", "regression with noise")
CATEGORY_METHODS = ("most frequent", "random draw", "nearest neighbours")
NEIGHBOURS = 5


def _dummies(table, categorical, drop_first):
    """One 0/1 column per category; a missing category is a category of its own."""
    if not categorical:
        return np.empty((len(table), 0))
    dummies = pd.get_dummies(table[categorical], dummy_na=True, drop_first=drop_first, dtype=float)
    return dummies.loc[:, dummies.to_numpy().any(axis=0)].to_numpy()


def regression_values(y, X, missing, rng=None):
    """Values for the missing entries of y from a linear regression on X, fitted where y is known.

    Without `rng` the predicted values. With `rng` one draw of plausible values: the residual
    variance and the coefficients are drawn from their posterior distribution under the usual
    non-informative prior, and normal noise of that variance is added to the prediction
    (van Buuren, 2018, algorithm 3.1).
    """
    known = ~missing
    q, r = np.linalg.qr(X[known])
    beta = np.linalg.solve(r, q.T @ y[known])
    if rng is None:
        return X[missing] @ beta
    residual = y[known] - X[known] @ beta
    sigma = np.sqrt(residual @ residual / rng.chisquare(known.sum() - X.shape[1]))
    beta = beta + sigma * np.linalg.solve(r, rng.standard_normal(X.shape[1]))
    return X[missing] @ beta + sigma * rng.standard_normal(missing.sum())


def chained_regression(table, numeric, categorical, rng=None, sweeps=5):
    """Numeric columns filled by regression on all other columns, one column after the other.

    Starts from the column means and repeats the round `sweeps` times when more than one column has
    missing values (chained equations; van Buuren and Groothuis-Oudshoorn, 2011).
    """
    lacking = table[numeric].isna()
    filled = table[numeric].fillna(table[numeric].mean())
    dummies = _dummies(table, categorical, drop_first=True)
    incomplete = [name for name in numeric if lacking[name].any()]
    for _ in range(sweeps if len(incomplete) > 1 else 1):
        for name in incomplete:
            others = filled[[other for other in numeric if other != name]].to_numpy()
            X = np.column_stack([np.ones(len(table)), others, dummies])
            hidden = lacking[name].to_numpy()
            filled.loc[hidden, name] = regression_values(filled[name].to_numpy(), X, hidden, rng)
    return filled


def fill_numeric(table, method, rng=None, numeric=NUMERIC, categorical=CATEGORICAL, group="Department"):
    """The numeric columns of the table with their missing values filled by one of METHODS."""
    values = table[numeric]
    if method == "mean":
        return values.fillna(values.mean())
    if method == "median":
        return values.fillna(values.median())
    if method == "group mean":
        return values.fillna(values.groupby(table[group]).transform("mean")).fillna(values.mean())
    if method == "nearest neighbours":
        mean, spread = values.mean(), values.std()
        scaled = np.column_stack([((values - mean) / spread).to_numpy(), _dummies(table, categorical, drop_first=False)])
        filled = KNNImputer(n_neighbors=NEIGHBOURS).fit_transform(scaled)[:, :len(numeric)]
        return pd.DataFrame(filled, index=table.index, columns=numeric) * spread + mean
    if method == "regression":
        return chained_regression(table, numeric, categorical)
    if method == "regression with noise":
        return chained_regression(table, numeric, categorical, rng)
    raise ValueError(f"unknown method: {method}")


def rubin(estimates, variances):
    """Pooled estimate, its total variance and degrees of freedom from several filled tables (Rubin, 1987)."""
    estimates, variances = np.asarray(estimates, float), np.asarray(variances, float)
    m = len(estimates)
    within, between = variances.mean(), estimates.var(ddof=1)
    total = within + (1 + 1 / m) * between
    dof = (m - 1) * (1 + within / ((1 + 1 / m) * between)) ** 2 if between > 0 else np.inf
    return float(estimates.mean()), float(total), float(dof)


def masking_experiment(table, rng, share=0.1, repeats=20, numeric=NUMERIC, categorical=CATEGORICAL):
    """Hide a share of the known numeric values, fill them with each method and compare with the truth.

    Returns the RMS error of the filled values per method and column, as a multiple of the standard
    deviation of the column.
    """
    known = table[numeric].notna().to_numpy()
    truth = table[numeric].to_numpy()
    squared = {method: np.zeros(len(numeric)) for method in METHODS}
    count = np.zeros(len(numeric))
    for _ in range(repeats):
        hidden = known & (rng.random(known.shape) < share)
        masked = table.copy()
        masked[numeric] = table[numeric].mask(hidden)
        count += hidden.sum(axis=0)
        for method in METHODS:
            filled = fill_numeric(masked, method, rng, numeric, categorical).to_numpy()
            squared[method] += np.where(hidden, (filled - truth) ** 2, 0.0).sum(axis=0)
    spread = table[numeric].std().to_numpy()
    return pd.DataFrame({method: np.sqrt(squared[method] / count) / spread for method in METHODS}, index=numeric).T


def fill_category(table, column, method, rng, numeric=NUMERIC, categorical=CATEGORICAL):
    """One categorical column with its missing values filled by one of CATEGORY_METHODS."""
    values = table[column]
    lacking = values.isna().to_numpy()
    filled = values.copy()
    if method == "most frequent":
        filled[lacking] = values.mode()[0]
    elif method == "random draw":
        filled[lacking] = rng.choice(values.dropna().to_numpy(), lacking.sum())
    elif method == "nearest neighbours":
        others = [name for name in categorical if name != column]
        scaled = (table[numeric] - table[numeric].mean()) / table[numeric].std()
        features = np.column_stack([scaled.fillna(0.0).to_numpy(), _dummies(table, others, drop_first=False)])
        model = KNeighborsClassifier(n_neighbors=3 * NEIGHBOURS).fit(features[~lacking], values[~lacking])
        filled[lacking] = model.predict(features[lacking])
    else:
        raise ValueError(f"unknown method: {method}")
    return filled


def category_experiment(table, rng, share=0.1, repeats=20, categorical=CATEGORICAL):
    """Hide a share of the known categories and fill them back.

    Returns, per column, the share filled correctly by each method, the largest category with its share
    of the known values, the chance that a random draw is right, and the share that the largest
    category has once every gap of the table is filled with it.
    """
    result = {}
    for column in categorical:
        known = table[column].notna().to_numpy()
        hits = {method: 0 for method in CATEGORY_METHODS}
        count = 0
        for _ in range(repeats):
            hidden = known & (rng.random(len(table)) < share)
            masked = table.copy()
            masked[column] = table[column].mask(hidden)
            count += hidden.sum()
            for method in CATEGORY_METHODS:
                hits[method] += (fill_category(masked, column, method, rng)[hidden] == table[column][hidden]).sum()
        shares = table[column].value_counts(normalize=True)
        top = shares.index[0]
        result[column] = {
            "most_frequent": top, "its_share": float(shares.iloc[0]), "chance_of_a_random_draw": float((shares**2).sum()),
            "its_share_after_filling_with_it": float(((table[column] == top) | table[column].isna()).mean()),
            "methods": {method: float(hits[method] / count) for method in CATEGORY_METHODS}}
    return result


def _estimates(salary, experience):
    """Mean, its naive variance, standard deviation, correlation with experience and slope on experience."""
    n = len(salary)
    spread = salary.std(ddof=1)
    covariance = np.cov(salary, experience)[0, 1]
    return np.array([salary.mean(), spread**2 / n, spread, covariance / (spread * experience.std(ddof=1)),
                     covariance / experience.var(ddof=1)])


def simulation(rng, n=1000, repeats=300, rate=0.3, imputations=20, truth_rows=1_000_000):
    """What each way of handling missing salaries does to estimates, on tables whose truth is known.

    Per mechanism and method, over `repeats` tables of `n` staff with a share `rate` of the salaries
    missing: RMS error of the filled values (multiple of the true standard deviation), mean, standard
    deviation, correlation with experience and slope on experience (each as the average estimate),
    and the share of 95 % intervals for the mean that contain the true mean.
    """
    numeric = ["Salary_NPR", "Experience_Years", "Age", "Rating"]
    population = staff_table(truth_rows, np.random.default_rng(0))
    truth = _estimates(population["Salary_NPR"].to_numpy(), population["Experience_Years"].to_numpy())
    methods = ("complete cases", "mean", "group mean", "nearest neighbours", "regression", "regression with noise",
               "multiple imputation")
    collected = {mechanism: {method: [] for method in methods} for mechanism in MECHANISMS}
    for _ in range(repeats):
        table = staff_table(n, rng)
        salary, experience = table["Salary_NPR"].to_numpy(), table["Experience_Years"].to_numpy()
        for mechanism in MECHANISMS:
            hidden = hide(table, mechanism, rate, rng)
            masked = table.copy()
            masked.loc[hidden, "Salary_NPR"] = np.nan
            for method in methods:
                error = np.nan
                if method == "complete cases":
                    values = _estimates(salary[~hidden], experience[~hidden])
                    dof = (~hidden).sum() - 1
                elif method == "multiple imputation":
                    draws = [_estimates(fill_numeric(masked, "regression with noise", rng, numeric, ["Department"])[
                        "Salary_NPR"].to_numpy(), experience) for _ in range(imputations)]
                    values = np.mean(draws, axis=0)
                    values[0], values[1], dof = rubin([draw[0] for draw in draws], [draw[1] for draw in draws])
                else:
                    filled = fill_numeric(masked, method, rng, numeric, ["Department"])["Salary_NPR"].to_numpy()
                    values = _estimates(filled, experience)
                    dof = n - 1
                    error = np.mean((filled[hidden] - salary[hidden]) ** 2)
                half = stats.t.ppf(0.975, dof) * np.sqrt(values[1])
                covered = abs(values[0] - truth[0]) <= half
                collected[mechanism][method].append([error, values[0], values[2], values[3], values[4], covered])
    result = {"truth": {"mean": float(truth[0]), "sd": float(truth[2]), "correlation": float(truth[3]), "slope": float(truth[4])},
              "settings": {"rows": n, "tables": repeats, "share_missing": rate, "imputations": imputations}}
    for mechanism in MECHANISMS:
        result[mechanism] = {}
        for method in methods:
            error, mean, spread, correlation, slope, covered = np.array(collected[mechanism][method], float).T
            result[mechanism][method] = {
                "rmse_in_sd": None if np.isnan(error).all() else float(np.sqrt(error.mean()) / truth[2]),
                "mean": float(mean.mean()), "mean_bias": float(mean.mean() - truth[0]),
                "sd_ratio": float(spread.mean() / truth[2]), "correlation": float(correlation.mean()),
                "slope_ratio": float(slope.mean() / truth[4]), "coverage": float(covered.mean())}
    return result
