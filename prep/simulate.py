"""A staff table in which the columns are related, and three ways in which values can go missing."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, special

DEPARTMENTS = {  # share of the staff and difference in salary (NPR per month)
    "Finance": (0.24, 10_000.0), "Operations": (0.21, 0.0), "Information Technology": (0.20, 20_000.0),
    "Human Resources": (0.18, -5_000.0), "Marketing": (0.17, -10_000.0)}
BASE_SALARY = 35_000.0
SALARY_PER_YEAR = 2_200.0
SALARY_NOISE = 15_000.0
MECHANISMS = ("MCAR", "MAR", "MNAR")


def staff_table(n, rng):
    """Staff whose experience is bounded by their age and whose salary depends on experience and department.

    Age 22 to 59; experience is a share of the years since the age of 21; salary is a base plus an
    amount per year of experience, plus the difference of the department, plus normal noise; the
    rating is unrelated to everything.
    """
    age = rng.integers(22, 60, n).astype(float)
    experience = np.round((age - 21) * rng.uniform(0.2, 1.0, n))
    names = list(DEPARTMENTS)
    department = rng.choice(names, n, p=[DEPARTMENTS[name][0] for name in names])
    offset = np.array([DEPARTMENTS[name][1] for name in department])
    salary = BASE_SALARY + SALARY_PER_YEAR * experience + offset + rng.normal(0.0, SALARY_NOISE, n)
    rating = np.round(rng.uniform(1.0, 5.0, n), 1)
    return pd.DataFrame({"Age": age, "Experience_Years": experience, "Department": department,
                         "Salary_NPR": salary, "Rating": rating})


def hide(table, mechanism, rate, rng, column="Salary_NPR", driver="Experience_Years", strength=1.5):
    """Rows whose value in `column` goes missing, as a boolean array.

    MCAR: every row with the same probability. MAR: the probability rises with the value of
    `driver`, which stays observed. MNAR: it rises with the value that goes missing. In the last
    two the log-odds change by `strength` per standard deviation, and the overall share is `rate`.
    """
    if mechanism == "MCAR":
        return rng.random(len(table)) < rate
    if mechanism not in ("MAR", "MNAR"):
        raise ValueError("mechanism must be 'MCAR', 'MAR' or 'MNAR'")
    values = table[driver if mechanism == "MAR" else column].to_numpy(float)
    z = (values - values.mean()) / values.std()
    shift = optimize.brentq(lambda a: special.expit(a + strength * z).mean() - rate, -30.0, 30.0)
    return rng.random(len(table)) < special.expit(shift + strength * z)
