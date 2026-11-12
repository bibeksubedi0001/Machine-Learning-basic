"""Checks of the table functions, the filling methods and the area functions against known answers."""
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import box
from shapely.ops import transform as apply

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prep import (METHODS, area_factor, category_experiment, chi_square_permutation, consistency, fill_numeric,
                  geodesic_area, hide, holm, load_table, masking_experiment, missing_summary, missingness_tests,
                  projected_area, regression_values, relation_tests, rubin, simulation, staff_table, to_lonlat,
                  transverse_mercator, web_mercator_area_factor)

TABLE = ROOT / "data" / "dataset_nepali.csv"


def square(longitude, latitude, side=1000.0):
    """A square of the given side in metres around a point, in longitude and latitude."""
    local = CRS.from_proj4(f"+proj=laea +lat_0={latitude} +lon_0={longitude} +ellps=WGS84")
    return to_lonlat(box(-side / 2, -side / 2, side / 2, side / 2), local)


class Table(unittest.TestCase):
    def test_table_of_the_repository(self):
        table = load_table(TABLE)
        summary = missing_summary(table)
        self.assertEqual(summary["rows"], 1000)
        self.assertEqual(summary["per_column"], {"Age": 30, "Salary_NPR": 30, "Experience_Years": 34, "Rating": 38,
                                                 "District": 41, "Department": 29})
        self.assertEqual(summary["rows_by_missing_values"], {0: 816, 1: 166, 2: 18})
        self.assertEqual(summary["complete_rows"], 816)
        facts = consistency(table)
        self.assertEqual(facts["experience_above_age"], 101)
        self.assertTrue(facts["join_dates_are_consecutive_days"])
        self.assertLess(facts["largest_rank_correlation_between_numeric_columns"], 0.1)

    def test_holm(self):
        np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        np.testing.assert_allclose(holm([0.5, 0.9]), [1.0, 1.0])

    def test_permuted_chi_square_tells_related_from_unrelated_columns(self):
        rng = np.random.default_rng(1)
        first = rng.integers(0, 3, 300)
        self.assertGreater(chi_square_permutation(first, rng.integers(0, 4, 300), rng, draws=500), 0.05)
        related = np.where(rng.random(300) < 0.6, first, rng.integers(0, 3, 300))
        self.assertLess(chi_square_permutation(first, related, rng, draws=500), 0.01)

    def test_missingness_that_depends_on_another_column_is_found(self):
        rng = np.random.default_rng(2)
        table = staff_table(1500, rng)
        table["District"] = rng.choice(["A", "B", "C"], len(table))
        for mechanism, found in (("MAR", True), ("MCAR", False)):
            masked = table.copy()
            masked.loc[hide(table, mechanism, 0.3, rng), "Salary_NPR"] = np.nan
            tests = missingness_tests(masked, rng).set_index(["missing_in", "other"])
            self.assertEqual(tests.loc[("Salary_NPR", "Experience_Years"), "p_holm"] < 0.01, found)
        relations = relation_tests(table, rng).set_index(["first", "second"])
        self.assertLess(relations.loc[("Salary_NPR", "Experience_Years"), "p_holm"], 1e-6)
        self.assertLess(relations.loc[("Salary_NPR", "Department"), "p_holm"], 1e-6)
        self.assertGreater(relations.loc[("Salary_NPR", "Rating"), "p_holm"], 0.05)


class Simulated(unittest.TestCase):
    def test_staff_table_obeys_its_rules(self):
        table = staff_table(20000, np.random.default_rng(3))
        self.assertTrue((table["Experience_Years"] <= table["Age"] - 21).all())
        self.assertTrue(table["Age"].between(22, 59).all())
        self.assertAlmostEqual(table["Salary_NPR"].corr(table["Experience_Years"]), 0.71, delta=0.02)
        self.assertAlmostEqual((table["Department"] == "Finance").mean(), 0.24, delta=0.015)
        self.assertLess(abs(table["Rating"].corr(table["Salary_NPR"])), 0.03)

    def test_hidden_share_and_what_it_depends_on(self):
        rng = np.random.default_rng(4)
        table = staff_table(20000, rng)
        gaps = {}
        for mechanism in ("MCAR", "MAR", "MNAR"):
            hidden = hide(table, mechanism, 0.3, rng)
            self.assertAlmostEqual(hidden.mean(), 0.3, delta=0.02)
            gaps[mechanism] = [table.loc[hidden, name].mean() - table.loc[~hidden, name].mean()
                               for name in ("Experience_Years", "Salary_NPR")]
        self.assertLess(abs(gaps["MCAR"][0]), 0.4)
        self.assertGreater(gaps["MAR"][0], 5.0)
        self.assertGreater(gaps["MNAR"][1], 15000.0)
        with self.assertRaises(ValueError):
            hide(table, "other", 0.3, rng)


class Filling(unittest.TestCase):
    def test_regression_gives_back_an_exact_line(self):
        x = np.arange(30.0)
        X = np.column_stack([np.ones(30), x])
        y = 2.0 + 3.0 * x
        missing = np.zeros(30, bool)
        missing[[4, 17, 29]] = True
        np.testing.assert_allclose(regression_values(y, X, missing), y[missing])
        np.testing.assert_allclose(regression_values(y, X, missing, np.random.default_rng(0)), y[missing], atol=1e-6)

    def test_regression_draws_scatter_around_the_prediction(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=400)
        X = np.column_stack([np.ones(400), x])
        y = 1.0 + 2.0 * x + rng.normal(0.0, 0.5, 400)
        missing = np.arange(400) < 100
        predicted = regression_values(y, X, missing)
        draws = np.array([regression_values(y, X, missing, rng) for _ in range(400)])
        np.testing.assert_allclose(draws.mean(axis=0), predicted, atol=0.12)
        self.assertAlmostEqual(draws.std(axis=0).mean(), 0.5, delta=0.05)

    def test_mean_filling_shrinks_the_spread_and_the_correlation(self):
        rng = np.random.default_rng(6)
        table = staff_table(20000, rng)
        true_sd = table["Salary_NPR"].std()
        true_r = table["Salary_NPR"].corr(table["Experience_Years"])
        table.loc[hide(table, "MCAR", 0.3, rng), "Salary_NPR"] = np.nan
        numeric = ["Salary_NPR", "Experience_Years", "Age", "Rating"]
        filled = fill_numeric(table, "mean", numeric=numeric, categorical=["Department"])
        self.assertAlmostEqual(filled["Salary_NPR"].std() / true_sd, np.sqrt(0.7), delta=0.01)
        self.assertAlmostEqual(filled["Salary_NPR"].corr(table["Experience_Years"]) / true_r, np.sqrt(0.7), delta=0.02)

    def test_every_method_fills_all_gaps_and_keeps_the_known_values(self):
        table = load_table(TABLE)
        known = table[["Age", "Salary_NPR", "Experience_Years", "Rating"]]
        for method in METHODS:
            filled = fill_numeric(table, method, np.random.default_rng(0))
            self.assertFalse(filled.isna().any().any(), method)
            np.testing.assert_allclose(filled.to_numpy()[known.notna().to_numpy()], known.to_numpy()[known.notna().to_numpy()],
                                       rtol=1e-9, err_msg=method)
        with self.assertRaises(ValueError):
            fill_numeric(table, "guess")

    def test_group_mean_and_nearest_neighbours_on_small_tables(self):
        table = pd.DataFrame({"y": [1.0, 3.0, np.nan, 10.0, 20.0, np.nan, np.nan], "g": ["a", "a", "a", "b", "b", "b", None]})
        filled = fill_numeric(table, "group mean", numeric=["y"], categorical=["g"], group="g")
        np.testing.assert_allclose(filled["y"], [1, 3, 2, 10, 20, 15, 8.5])
        x = np.arange(20.0)
        table = pd.DataFrame({"y": 10.0 * x, "x": x})
        table.loc[0, "y"] = np.nan
        filled = fill_numeric(table, "nearest neighbours", numeric=["y", "x"], categorical=[])
        self.assertAlmostEqual(filled.loc[0, "y"], 30.0)                # the mean of the rows with x = 1 to 5

    def test_rubin_rules(self):
        estimate, total, dof = rubin([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
        self.assertAlmostEqual(estimate, 2.0)
        self.assertAlmostEqual(total, 1.0 + 4.0 / 3.0)
        self.assertAlmostEqual(dof, 2 * 1.75**2)
        self.assertEqual(rubin([2.0, 2.0], [1.0, 3.0]), (2.0, 2.0, np.inf))

    def test_on_unrelated_columns_no_method_beats_the_mean(self):
        rng = np.random.default_rng(7)
        table = pd.DataFrame(rng.uniform(size=(600, 4)), columns=["Age", "Salary_NPR", "Experience_Years", "Rating"])
        table["District"] = rng.choice(list("abcd"), 600)
        table["Department"] = rng.choice(list("xyz"), 600, p=[0.5, 0.3, 0.2])
        errors = masking_experiment(table, rng, share=0.1, repeats=6).mean(axis=1)
        self.assertAlmostEqual(errors["mean"], 1.0, delta=0.06)
        self.assertAlmostEqual(errors["nearest neighbours"], np.sqrt(1.2), delta=0.08)
        self.assertAlmostEqual(errors["regression with noise"], np.sqrt(2.0), delta=0.12)
        self.assertGreater(errors["regression"], errors["mean"] - 0.01)
        categories = category_experiment(table, rng, repeats=6)["Department"]
        self.assertEqual(categories["most_frequent"], "x")
        self.assertEqual(categories["its_share_after_filling_with_it"], categories["its_share"])       # the table has no gaps
        self.assertAlmostEqual(categories["methods"]["most frequent"], categories["its_share"], delta=0.08)
        self.assertAlmostEqual(categories["methods"]["random draw"], categories["chance_of_a_random_draw"], delta=0.08)
        table.loc[table.index[:60], "Department"] = None
        after = category_experiment(table, rng, repeats=1)["Department"]
        self.assertAlmostEqual(after["its_share_after_filling_with_it"], 0.9 * after["its_share"] + 0.1, places=9)

    def test_simulation_shows_the_known_effects(self):
        result = simulation(np.random.default_rng(8), n=500, repeats=12, imputations=5, truth_rows=200_000)
        self.assertAlmostEqual(result["truth"]["correlation"], 0.71, delta=0.01)
        mcar, mnar = result["MCAR"], result["MNAR"]
        self.assertAlmostEqual(mcar["mean"]["sd_ratio"], np.sqrt(0.7), delta=0.03)
        self.assertAlmostEqual(mcar["mean"]["rmse_in_sd"], 1.0, delta=0.06)
        self.assertLess(mcar["regression"]["rmse_in_sd"], 0.7)
        self.assertGreater(mcar["regression with noise"]["rmse_in_sd"], mcar["regression"]["rmse_in_sd"])
        self.assertAlmostEqual(mcar["regression with noise"]["sd_ratio"], 1.0, delta=0.03)
        self.assertIsNone(mcar["multiple imputation"]["rmse_in_sd"])
        self.assertLess(result["MAR"]["complete cases"]["mean_bias"], -3000.0)
        self.assertLess(abs(result["MAR"]["regression"]["mean_bias"]), 1500.0)
        self.assertLess(mnar["multiple imputation"]["mean_bias"], -2000.0)


class Areas(unittest.TestCase):
    def test_geodesic_area_of_a_square_kilometre(self):
        for longitude, latitude in ((85.0, 28.0), (10.0, 60.0), (-70.0, -15.0)):
            self.assertAlmostEqual(geodesic_area(square(longitude, latitude)) / 1e6, 1.0, places=6)

    def test_equal_area_projection_keeps_the_area_and_web_mercator_follows_its_formula(self):
        for latitude in (0.0, 28.0, 60.0):
            cell = square(85.0, latitude)
            self.assertAlmostEqual(projected_area(cell, "EPSG:6933") / geodesic_area(cell), 1.0, places=5)
            self.assertAlmostEqual(projected_area(cell, "EPSG:3857") / geodesic_area(cell) / web_mercator_area_factor(latitude),
                                   1.0, places=5)
        self.assertAlmostEqual(web_mercator_area_factor(60.0), 4.0, delta=0.03)

    def test_transverse_mercator_factors(self):
        self.assertAlmostEqual(area_factor("EPSG:32645", 87.0, 28.0), 0.9996**2, places=8)
        self.assertAlmostEqual(area_factor(transverse_mercator(84, 0.9999), 84.0, 28.0), 0.9999**2, places=8)
        cell = square(85.5, 27.5)
        self.assertAlmostEqual(projected_area(cell, transverse_mercator(87)) / projected_area(cell, "EPSG:32645"), 1.0, places=9)
        self.assertAlmostEqual(projected_area(cell, "EPSG:32645") / geodesic_area(cell) / area_factor("EPSG:32645", 85.5, 27.5),
                               1.0, places=6)
        self.assertGreater(area_factor("EPSG:32645", 80.0, 29.0), 1.01)

    def test_polygon_of_the_repository(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import geopandas as gpd
            shape = gpd.read_file(ROOT / "data" / "new_shapefile.shp")
        self.assertEqual(shape.crs.to_epsg(), 24047)
        polygon = to_lonlat(shape.geometry.iloc[0], shape.crs)
        self.assertTrue(84.90 < polygon.centroid.x < 84.93 and 27.57 < polygon.centroid.y < 27.60)
        self.assertAlmostEqual(shape.geometry.iloc[0].area / geodesic_area(polygon), 1.048, delta=0.001)
        again = apply(Transformer.from_crs("EPSG:4326", shape.crs, always_xy=True).transform, polygon)
        self.assertAlmostEqual(again.area / shape.geometry.iloc[0].area, 1.0, places=7)


if __name__ == "__main__":
    unittest.main()
