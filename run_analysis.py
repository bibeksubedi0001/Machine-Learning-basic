"""Two studies: filling the missing values of a staff table, and areas in map coordinate systems."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely

with warnings.catch_warnings():
    warnings.simplefilter("ignore")                 # geopandas 0.14 warns about its own future defaults on import
    import geopandas as gpd

from prep import (CATEGORICAL, CATEGORY_METHODS, MECHANISMS, METHODS, NUMERIC, area_factor, category_experiment, consistency,
                  geodesic_area, load_table, masking_experiment, missing_summary, missingness_tests, projected_area,
                  relation_tests, simulation, to_lonlat, transverse_mercator, web_mercator_area_factor)
from prep.impute import NEIGHBOURS

# Liberation Serif has the metrics of Times New Roman and stands in where that font is not installed
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"]})

ROOT = Path(__file__).resolve().parent
DATA, FIGURES, RESULTS = ROOT / "data", ROOT / "figures", ROOT / "results"
SEED = 2026
MASKED_SHARE, MASKING_REPEATS = 0.1, 20
NEPAL_LONGITUDES = (80.06, 88.20)                  # western and eastern end of the country
NEPAL_LATITUDE = 28.0
VERTEX_SPACING = 0.1                               # degrees; outlines are given vertices at least this close
WEB_MERCATOR_LIMIT = 85.05                         # degrees of latitude that Web Mercator shows
MECHANISM_STYLE = {"MCAR": ("tab:blue", "o"), "MAR": ("tab:orange", "s"), "MNAR": ("tab:red", "^")}
# countries named in the figure, with the offset of the label in points (to the left of the point when negative)
LABELLED = {"Nepal": (4, -10), "India": (4, -11), "Brazil": (4, -10), "Russia": (5, -10), "Canada": (-5, 5),
            "Norway": (-6, 3), "Greenland": (-6, 2)}
LONGER_NAMES = {"Norway": "Norway with Svalbard"}  # what the outline of that name contains
UTM_EXAMPLES = ("Nepal", "India", "China", "Germany", "Australia", "Brazil")


def finish(fig, name):
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def percent(value, digits=2):
    """A signed percentage; a value that rounds to zero is written without a sign."""
    rounded = round(100 * value, digits)
    return f"{rounded:+.{digits}f} %" if rounded else f"{0:.{digits}f} %"


# ---------------------------------------------------------------------------------------------- the table

def analyse_table(table, rng):
    relations = relation_tests(table, rng)
    missingness = missingness_tests(table, rng)
    lowest = missingness.loc[missingness["p"].idxmin()]
    return {
        "missing": missing_summary(table), "consistency": consistency(table),
        "relations_between_columns": {
            "tests": int(len(relations)), "smallest_p": float(relations["p"].min()),
            "smallest_holm_p": float(relations["p_holm"].min())},
        "missingness_against_other_columns": {
            "tests": int(len(missingness)), "p_below_0.05": int((missingness["p"] < 0.05).sum()),
            "smallest_p": float(lowest["p"]), "smallest_p_is_for": f"{lowest['missing_in']} missing against {lowest['other']}",
            "smallest_holm_p": float(missingness["p_holm"].min())},
    }


def plot_missing(table, summary):
    columns = NUMERIC + CATEGORICAL
    lacking = table[columns].isna().to_numpy()
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    rows, cols = np.nonzero(lacking)
    left.scatter(rows + 1, cols, marker="|", s=160, color="black", linewidths=0.8)
    left.set_yticks(range(len(columns)), columns)
    left.set(xlabel="row of the table", xlim=(0, len(table) + 1), ylim=(len(columns) - 0.5, -0.5),
             title=f"The {int(lacking.sum())} missing values")
    counts = summary["missing"]["per_column"]
    bars = right.barh(columns, [counts[name] for name in columns], color="0.45")
    right.bar_label(bars, padding=2, fontsize=9)
    right.invert_yaxis()
    by_row = summary["missing"]["rows_by_missing_values"]
    right.set(xlabel="missing values", xlim=(0, 1.15 * max(counts.values())),
              title="Rows lacking 0, 1 and 2 values: " + ", ".join(str(by_row.get(k, 0)) for k in (0, 1, 2)))
    fig.tight_layout()
    finish(fig, "table_missing.png")


def plot_masking(masking, categories):
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.7, 1]})
    width = 0.8 / len(NUMERIC)
    for position, name in enumerate(NUMERIC):
        left.bar(np.arange(len(METHODS)) + (position - (len(NUMERIC) - 1) / 2) * width, masking[name], width, label=name)
    expected = {"mean": 1.0, "median": 1.0, "group mean": 1.0, "nearest neighbours": np.sqrt(1 + 1 / NEIGHBOURS),
                "regression": 1.0, "regression with noise": np.sqrt(2.0)}
    for position, method in enumerate(METHODS):
        line = left.hlines(expected[method], position - 0.45, position + 0.45, color="black", linestyles="--", linewidth=1)
    line.set_label("expected for unrelated columns")
    left.set_xticks(range(len(METHODS)), [method.replace(" ", "\n", 1) for method in METHODS])
    left.set(ylabel="RMS error of the filled values / standard deviation", ylim=(0, 1.7),
             title="Numeric columns of the table: hidden values against filled values")
    left.grid(axis="y", alpha=0.3)
    left.legend(fontsize=8, ncol=3, loc="upper left")

    width = 0.8 / len(CATEGORICAL)
    for position, name in enumerate(CATEGORICAL):
        shares = [100 * categories[name]["methods"][method] for method in CATEGORY_METHODS]
        centre = np.arange(len(CATEGORY_METHODS)) + (position - (len(CATEGORICAL) - 1) / 2) * width
        bars = right.bar(centre, shares, width, label=f"{name} (largest group {100 * categories[name]['its_share']:.0f} %)")
        right.bar_label(bars, fmt="%.0f", fontsize=9)
    right.set_xticks(range(len(CATEGORY_METHODS)), [method.replace(" ", "\n", 1) for method in CATEGORY_METHODS])
    right.set(ylabel="hidden categories filled correctly (%)", ylim=(0, 40), title="Categorical columns of the table")
    right.grid(axis="y", alpha=0.3)
    right.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    finish(fig, "table_masking.png")


def plot_simulation(result):
    methods = list(result["MCAR"])
    truth = result["truth"]
    panels = [("rmse_in_sd", "RMS error of the filled salaries\n(true standard deviation = 1)", None),
              ("mean_bias", "Mean salary: estimate minus truth (NPR)", 0.0),
              ("sd_ratio", "Standard deviation of salary:\nestimate / truth", 1.0),
              ("correlation", "Correlation of salary with experience", truth["correlation"]),
              ("slope_ratio", "Slope of salary on experience:\nestimate / truth", 1.0),
              ("coverage", "95 % intervals for the mean salary\nthat contain the truth (%)", 95.0)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for ax, (key, title, reference) in zip(axes.ravel(), panels):
        if reference is not None:
            ax.axvline(reference, color="0.4", linestyle="--", linewidth=1)
        for mechanism in MECHANISMS:
            colour, marker = MECHANISM_STYLE[mechanism]
            values = [result[mechanism][method][key] for method in methods]
            shown = [(100 * value if key == "coverage" else value, position) for position, value in enumerate(values)
                     if value is not None]
            ax.plot(*zip(*shown), marker, color=colour, markersize=6, alpha=0.85, label=mechanism)
        ax.set_yticks(range(len(methods)), methods)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0, 0].invert_yaxis()
    axes[0, 0].legend(title="values go missing", fontsize=8, title_fontsize=8, loc="lower right")
    fig.tight_layout()
    finish(fig, "simulation.png")


# ---------------------------------------------------------------------------------------------- the areas

def analyse_polygon():
    shape = gpd.read_file(DATA / "new_shapefile.shp")
    stored = shape.geometry.iloc[0]
    polygon = to_lonlat(stored, shape.crs)
    truth = geodesic_area(polygon)
    centre = polygon.centroid
    meridian = next(p.value for p in shape.crs.coordinate_operation.params if p.name == "Longitude of natural origin")
    zone = 3 * round(centre.x / 3)                  # central meridian of the 3-degree zone that holds the polygon
    systems = [(f"as stored: {shape.crs.name}", shape.crs), ("WGS 84 / UTM zone 44N", "EPSG:32644"),
               ("WGS 84 / UTM zone 45N", "EPSG:32645"),
               (f"transverse Mercator, central meridian {zone} °E, scale 0.9999", transverse_mercator(zone, 0.9999)),
               ("Web Mercator", "EPSG:3857"), ("equal-area cylindrical (EPSG:6933)", "EPSG:6933")]
    rows = [{"system": name, "area_m2": float(projected_area(polygon, crs)), "ratio": float(projected_area(polygon, crs) / truth)}
            for name, crs in systems]
    local = f"+proj=laea +lat_0={centre.y} +lon_0={centre.x} +ellps=WGS84"
    outline = gpd.GeoSeries([polygon], crs=4326).to_crs(local).iloc[0]
    facts = {"file_crs": f"EPSG:{shape.crs.to_epsg()} ({shape.crs.name})", "vertices": len(stored.exterior.coords),
             "longitude": float(centre.x), "latitude": float(centre.y), "geodesic_area_m2": float(truth),
             "degrees_from_the_central_meridian_of_the_file_crs": float(meridian - centre.x),
             "area_factor_of_the_file_crs_at_the_polygon": float(area_factor(shape.crs, centre.x, centre.y)), "systems": rows}
    return facts, outline


def analyse_countries():
    world = gpd.read_file(DATA / "countries_110m.gpkg")
    names = world["name"].to_numpy()
    # with close vertices an edge is the same line in every coordinate system
    outlines = [shapely.segmentize(geometry, VERTEX_SPACING) for geometry in world.geometry]
    truth = np.array([geodesic_area(outline) for outline in outlines])
    latitude = np.array([outline.centroid.y for outline in outlines])
    per_square_degree = truth / np.array([outline.area for outline in outlines]) / 1e6
    with np.errstate(invalid="ignore"):             # some outlines cannot be projected to UTM zone 45N
        ratios = {name: np.array([projected_area(outline, crs) for outline in outlines]) / truth
                  for name, crs in (("Web Mercator", "EPSG:3857"), ("equal-area cylindrical", "EPSG:6933"),
                                    ("UTM zone 45N", "EPSG:32645"))}
    shown = np.array([-WEB_MERCATOR_LIMIT <= outline.bounds[1] and outline.bounds[3] <= WEB_MERCATOR_LIMIT for outline in outlines])
    mercator = np.where(shown, ratios["Web Mercator"], np.nan)
    utm = ratios["UTM zone 45N"]
    computed = np.isfinite(utm)
    close = computed & (np.abs(utm - 1) < 0.01)

    def of(values, name):
        return float(values[names == name][0])

    facts = {
        "countries": int(len(world)), "source": "Natural Earth 1:110m", "vertex_spacing_deg": VERTEX_SPACING,
        "km2_per_square_degree": {"smallest": float(per_square_degree.min()), "largest": float(per_square_degree.max()),
                                  "Nepal": of(per_square_degree, "Nepal")},
        "web_mercator": {
            "left_out": names[~shown].tolist(), "smallest_ratio": float(np.nanmin(mercator)),
            "smallest_is": str(names[np.nanargmin(mercator)]), "median_ratio": float(np.nanmedian(mercator)),
            "largest_ratio": float(np.nanmax(mercator)), "largest_is": str(names[np.nanargmax(mercator)]),
            "examples": {LONGER_NAMES.get(name, name): of(mercator, name) for name in LABELLED}},
        "equal_area_cylindrical": {"largest_departure_from_1": float(np.abs(ratios["equal-area cylindrical"] - 1).max())},
        "utm_zone_45n": {
            "within_1_percent": names[close].tolist(), "off_by_more_than_half": int((computed & (np.abs(utm - 1) > 0.5)).sum()),
            "not_computable": names[~computed].tolist(), "largest_ratio": float(np.nanmax(utm)),
            "largest_is": str(names[np.nanargmax(utm)]), "examples": {name: of(utm, name) for name in UTM_EXAMPLES}},
    }
    return facts, names, latitude, mercator


def analyse_zones():
    longitude = np.round(np.arange(80.0, 88.2501, 0.05), 2)
    systems = {"UTM zone 44N": "EPSG:32644", "UTM zone 45N": "EPSG:32645"}
    curves = {name: np.array([area_factor(crs, lon, NEPAL_LATITUDE) - 1 for lon in longitude]) for name, crs in systems.items()}
    meridian = 3 * np.round(longitude / 3)
    curves["3° zones, scale 0.9999"] = np.array([area_factor(transverse_mercator(cm, 0.9999), lon, NEPAL_LATITUDE) - 1
                                                 for lon, cm in zip(longitude, meridian)])
    west, east = NEPAL_LONGITUDES
    own_zone = np.where(longitude < 84.0, curves["UTM zone 44N"], curves["UTM zone 45N"])
    inside = (longitude >= west) & (longitude <= east)

    def at(name, lon):
        return float(area_factor(systems[name], lon, NEPAL_LATITUDE) - 1)

    facts = {
        "latitude": NEPAL_LATITUDE, "longitudes": list(NEPAL_LONGITUDES),
        "utm_44n": {"at_the_west_end": at("UTM zone 44N", west), "at_84_e": at("UTM zone 44N", 84.0), "at_the_east_end": at("UTM zone 44N", east)},
        "utm_45n": {"at_the_west_end": at("UTM zone 45N", west), "at_84_e": at("UTM zone 45N", 84.0), "at_the_east_end": at("UTM zone 45N", east)},
        "utm_zone_by_longitude": {"lowest": float(own_zone[inside].min()), "highest": float(own_zone[inside].max())},
        "three_degree_zones": {"lowest": float(curves["3° zones, scale 0.9999"][inside].min()),
                               "highest": float(curves["3° zones, scale 0.9999"][inside].max())},
        "web_mercator": float(web_mercator_area_factor(NEPAL_LATITUDE) - 1),
    }
    return facts, longitude, curves


def plot_polygon(facts, outline):
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
    x, y = outline.exterior.xy
    left.fill(x, y, color="0.85", edgecolor="black", linewidth=1)
    left.set_aspect("equal")
    left.set(xlabel="east of the centre (m)", ylabel="north of the centre (m)",
             title=f"The polygon: {facts['geodesic_area_m2'] / 1e4:.2f} ha at {facts['longitude']:.3f} °E, {facts['latitude']:.3f} °N")
    left.grid(alpha=0.3)
    names = [row["system"].replace(": ", ":\n", 1).replace(", central", ",\ncentral", 1).replace(" (EPSG", "\n(EPSG", 1)
             for row in facts["systems"]]
    errors = [row["ratio"] - 1 for row in facts["systems"]]
    right.barh(names, [100 * error for error in errors], color=["tab:red" if abs(error) > 0.01 else "0.45" for error in errors])
    for position, error in enumerate(errors):
        right.annotate(percent(error).replace("-", "\N{MINUS SIGN}"), (max(100 * error, 0.0), position), xytext=(4, 0),
                       textcoords="offset points", va="center", fontsize=9)
    right.invert_yaxis()
    right.set(xlabel="area in the coordinate system minus area on the ellipsoid (%)", xlim=(-1, 118 * max(errors)),
              title="Area of the polygon by coordinate system")
    right.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    finish(fig, "polygon_area.png")


def plot_countries(names, latitude, mercator):
    shown = np.isfinite(mercator)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.scatter(np.abs(latitude[shown]), mercator[shown], s=14, color="tab:blue", alpha=0.7, linewidths=0, label="country")
    grid = np.linspace(0, 80, 161)
    ax.plot(grid, web_mercator_area_factor(grid), color="black", linewidth=1, label="factor at one latitude")
    for name, offset in LABELLED.items():
        index = np.flatnonzero(names == name)[0]
        ax.annotate(LONGER_NAMES.get(name, name), (abs(latitude[index]), mercator[index]), xytext=offset,
                    textcoords="offset points", fontsize=8, ha="right" if offset[0] < 0 else "left")
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
    ax.set(xlabel="latitude of the centre of the country, north or south (°)", ylabel="area in Web Mercator / area on the ellipsoid",
           xlim=(0, 80), ylim=(0.9, 30), title="Web Mercator enlarges areas away from the equator")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    finish(fig, "countries_web_mercator.png")


def plot_zones(longitude, curves, polygon_longitude):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    styles = {"UTM zone 44N": ("tab:blue", "-"), "UTM zone 45N": ("tab:red", "-"), "3° zones, scale 0.9999": ("black", "--")}
    for name, values in curves.items():
        ax.plot(longitude, 100 * values, color=styles[name][0], linestyle=styles[name][1], label=name)
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.axvline(84.0, color="0.5", linestyle=":", linewidth=1)
    for text, side, offset in (("zone 44N", "right", -5), ("zone 45N", "left", 5)):
        ax.annotate(text, (84.0, 0.5), xycoords=("data", "axes fraction"), xytext=(offset, 0), textcoords="offset points",
                    ha=side, fontsize=8, color="0.3")
    ax.axvline(polygon_longitude, color="tab:green", linewidth=1)
    ax.annotate("the polygon", (polygon_longitude, 0.66), xycoords=("data", "axes fraction"), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color="tab:green")
    ax.set(xlabel=f"longitude (°E) at latitude {NEPAL_LATITUDE:g} °N; Nepal reaches from {NEPAL_LONGITUDES[0]} to {NEPAL_LONGITUDES[1]} °E",
           ylabel="area error (%)", xlim=(80, 88.25), title="Area error of transverse Mercator systems across Nepal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.36, 1.0))
    fig.tight_layout()
    finish(fig, "nepal_zones.png")


# ---------------------------------------------------------------------------------------------- the report

def report(summary, masking):
    table, sim, polygon, world, zones = (summary[key] for key in ("table", "simulation", "polygon", "countries", "nepal_zones"))
    missing, facts = table["missing"], table["consistency"]
    relations, dependence = table["relations_between_columns"], table["missingness_against_other_columns"]
    lines = ["# Results", "", "Command: `python run_analysis.py`", "", "## The staff table", "",
             f"{missing['rows']} rows; {missing['complete_rows']} are complete, "
             + ", ".join(f"{count} lack {k}" for k, count in missing["rows_by_missing_values"].items() if k) + " values.", "",
             "| Column | Missing values |", "|---|---|"]
    lines += [f"| {name} | {count} |" for name, count in missing["per_column"].items()]
    lines += ["", f"- Relations between the six columns: {relations['tests']} tests, smallest p-value {relations['smallest_p']:.3f}, "
              f"{relations['smallest_holm_p']:.2f} after Holm's adjustment.",
              f"- Largest rank correlation between two numeric columns: {facts['largest_rank_correlation_between_numeric_columns']:.3f}.",
              "- Excess kurtosis of the numeric columns (a uniform distribution has -1.2): "
              + ", ".join(f"{name} {value:.2f}" for name, value in facts["excess_kurtosis"].items()) + ".",
              f"- Experience above age: {facts['experience_above_age']} of {facts['rows_with_age_and_experience']} rows; "
              f"experience above age minus 15: {facts['experience_above_age_minus_15']}.",
              f"- Join dates: {'one per day without a gap' if facts['join_dates_are_consecutive_days'] else 'not consecutive'}, "
              f"{facts['first_join_date']} to {facts['last_join_date']}.",
              f"- Missing values against the values of the other columns: {dependence['tests']} tests, {dependence['p_below_0.05']} with "
              f"p below 0.05, smallest {dependence['smallest_p']:.3f} ({dependence['smallest_p_is_for']}), "
              f"{dependence['smallest_holm_p']:.2f} after Holm's adjustment.",
              "", f"### Hidden values filled back ({100 * MASKED_SHARE:.0f} % of the known values, {MASKING_REPEATS} repeats)", "",
              "RMS error of the filled values as a multiple of the standard deviation of the column:", "",
              "| Method | " + " | ".join(NUMERIC) + " |", "|---|" + "---|" * len(NUMERIC)]
    lines += [f"| {method} | " + " | ".join(f"{masking.loc[method, name]:.3f}" for name in NUMERIC) + " |" for method in METHODS]
    lines += ["", "Share of the hidden categories filled correctly:", "",
              "| Column | " + " | ".join(CATEGORY_METHODS) + " | Largest group | Its share of the known values | "
              "Its share after every gap is filled with it |", "|---|" + "---|" * (len(CATEGORY_METHODS) + 3)]
    lines += [f"| {name} | " + " | ".join(f"{100 * row['methods'][method]:.1f} %" for method in CATEGORY_METHODS)
              + f" | {row['most_frequent']} | {100 * row['its_share']:.1f} % | {100 * row['its_share_after_filling_with_it']:.1f} % |"
              for name, row in summary["categories"].items()]
    settings, truth = sim["settings"], sim["truth"]
    lines += ["", "## Simulated staff tables", "",
              f"{settings['tables']} tables of {settings['rows']} rows; {100 * settings['share_missing']:.0f} % of the salaries missing; "
              f"{settings['imputations']} filled tables per multiple imputation. Truth: mean salary {truth['mean']:,.0f} NPR, standard "
              f"deviation {truth['sd']:,.0f} NPR, correlation with experience {truth['correlation']:.3f}, slope {truth['slope']:,.0f} NPR per year."]
    for mechanism in MECHANISMS:
        lines += ["", f"### {mechanism}", "",
                  "| Method | RMS error of the filled values / SD | Mean minus truth (NPR) | SD / truth | Correlation | Slope / truth | "
                  "95 % intervals containing the true mean |", "|---|---|---|---|---|---|---|"]
        for method, row in sim[mechanism].items():
            error = "-" if row["rmse_in_sd"] is None else f"{row['rmse_in_sd']:.2f}"
            lines.append(f"| {method} | {error} | {row['mean_bias']:+,.0f} | {row['sd_ratio']:.3f} | {row['correlation']:.3f} | "
                         f"{row['slope_ratio']:.3f} | {100 * row['coverage']:.0f} % |")
    lines += ["", "## Area of the polygon", "",
              f"`data/new_shapefile.shp`: one polygon of {polygon['vertices']} vertices, stored in {polygon['file_crs']}, at "
              f"{polygon['longitude']:.4f} °E, {polygon['latitude']:.4f} °N, "
              f"{polygon['degrees_from_the_central_meridian_of_the_file_crs']:.1f}° west of the central meridian of that system, "
              f"where its area factor is {polygon['area_factor_of_the_file_crs_at_the_polygon']:.4f}. "
              f"Area on the WGS 84 ellipsoid: {polygon['geodesic_area_m2']:,.0f} m².", "",
              "| Coordinate system | Area (m²) | Difference from the area on the ellipsoid |", "|---|---|---|"]
    lines += [f"| {row['system']} | {row['area_m2']:,.0f} | {percent(row['ratio'] - 1)} |" for row in polygon["systems"]]
    mercator, utm = world["web_mercator"], world["utm_zone_45n"]
    square = world["km2_per_square_degree"]
    lines += ["", "## Areas of countries", "",
              f"{world['countries']} outlines ({world['source']}), with vertices added so that none are more than "
              f"{world['vertex_spacing_deg']:g}° apart. Each area is divided by the area of the same outline on the ellipsoid.", "",
              f"- One square degree is between {square['smallest']:,.0f} and {square['largest']:,.0f} km², depending on the country; "
              f"{square['Nepal']:,.0f} km² for Nepal.",
              f"- Web Mercator: between {mercator['smallest_ratio']:.3f} ({mercator['smallest_is']}) and "
              f"{mercator['largest_ratio']:.1f} ({mercator['largest_is']}), median {mercator['median_ratio']:.2f}. "
              + ", ".join(f"{name} {value:.2f}" for name, value in mercator["examples"].items())
              + f". Left out, because it reaches beyond the latitudes that the projection shows: {', '.join(mercator['left_out'])}.",
              f"- Equal-area cylindrical projection (EPSG:6933): within "
              f"{100 * world['equal_area_cylindrical']['largest_departure_from_1']:.4f} % for every country.",
              f"- UTM zone 45N used for every country: within 1 % for {len(utm['within_1_percent'])} "
              f"({', '.join(utm['within_1_percent'])}); off by more than half for {utm['off_by_more_than_half']}; not computable for "
              f"{len(utm['not_computable'])} ({', '.join(utm['not_computable'])}); largest {utm['largest_ratio']:.0f} "
              f"({utm['largest_is']}). " + ", ".join(f"{name} {value:.3f}" for name, value in utm["examples"].items()) + ".",
              "", f"## Transverse Mercator systems across Nepal (latitude {zones['latitude']:g} °N)", "",
              "| System | Area error |", "|---|---|"]
    west, east = zones["longitudes"]
    for name, key in (("UTM zone 44N", "utm_44n"), ("UTM zone 45N", "utm_45n")):
        row = zones[key]
        lines.append(f"| {name} | {percent(row['at_the_west_end'])} at {west} °E, {percent(row['at_84_e'])} at 84 °E, "
                     f"{percent(row['at_the_east_end'])} at {east} °E |")
    lines += [f"| UTM, zone chosen by longitude | {percent(zones['utm_zone_by_longitude']['lowest'])} to "
              f"{percent(zones['utm_zone_by_longitude']['highest'])} |",
              f"| 3° zones with central meridians 81, 84 and 87 °E and scale 0.9999 | {percent(zones['three_degree_zones']['lowest'])} to "
              f"{percent(zones['three_degree_zones']['highest'])} |",
              f"| Web Mercator | {percent(zones['web_mercator'], 1)} |", ""]
    return "\n".join(lines)


def main():
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    table = load_table(DATA / "dataset_nepali.csv")
    summary = {"seed": SEED, "table": analyse_table(table, rng)}
    masking = masking_experiment(table, rng, MASKED_SHARE, MASKING_REPEATS)
    summary["masking"] = {method: masking.loc[method].to_dict() for method in METHODS}
    summary["categories"] = category_experiment(table, rng, MASKED_SHARE, MASKING_REPEATS)
    summary["simulation"] = simulation(rng)

    polygon, outline = analyse_polygon()
    countries, names, latitude, mercator = analyse_countries()
    zones, longitude, curves = analyse_zones()
    summary.update({"polygon": polygon, "countries": countries, "nepal_zones": zones})

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (RESULTS / "summary.md").write_text(report(summary, masking), encoding="utf-8")
    plot_missing(table, summary["table"])
    plot_masking(masking, summary["categories"])
    plot_simulation(summary["simulation"])
    plot_polygon(polygon, outline)
    plot_countries(names, latitude, mercator)
    plot_zones(longitude, curves, polygon["longitude"])
    print(f"wrote {RESULTS / 'summary.md'} and the figures in {FIGURES}")


if __name__ == "__main__":
    main()
