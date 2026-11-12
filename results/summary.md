# Results

Command: `python run_analysis.py`

## The staff table

1000 rows; 816 are complete, 166 lack 1, 18 lack 2 values.

| Column | Missing values |
|---|---|
| Age | 30 |
| Salary_NPR | 30 |
| Experience_Years | 34 |
| Rating | 38 |
| District | 41 |
| Department | 29 |

- Relations between the six columns: 15 tests, smallest p-value 0.040, 0.60 after Holm's adjustment.
- Largest rank correlation between two numeric columns: 0.067.
- Excess kurtosis of the numeric columns (a uniform distribution has -1.2): Age -1.21, Salary_NPR -1.26, Experience_Years -1.16, Rating -1.15.
- Experience above age: 101 of 937 rows; experience above age minus 15: 338.
- Join dates: one per day without a gap, 2000-01-01 to 2002-09-26.
- Missing values against the values of the other columns: 30 tests, 3 with p below 0.05, smallest 0.005 (Department missing against Experience_Years), 0.16 after Holm's adjustment.

### Hidden values filled back (10 % of the known values, 20 repeats)

RMS error of the filled values as a multiple of the standard deviation of the column:

| Method | Age | Salary_NPR | Experience_Years | Rating |
|---|---|---|---|---|
| mean | 1.007 | 1.003 | 0.997 | 1.019 |
| median | 1.006 | 1.005 | 0.999 | 1.018 |
| group mean | 1.012 | 1.008 | 1.000 | 1.023 |
| nearest neighbours | 1.089 | 1.082 | 1.075 | 1.119 |
| regression | 1.021 | 1.012 | 1.003 | 1.036 |
| regression with noise | 1.421 | 1.396 | 1.408 | 1.422 |

Share of the hidden categories filled correctly:

| Column | most frequent | random draw | nearest neighbours | Largest group | Its share of the known values | Its share after every gap is filled with it |
|---|---|---|---|---|---|---|
| District | 13.0 % | 12.3 % | 14.5 % | Bhaktapur | 13.8 % | 17.3 % |
| Department | 24.5 % | 20.3 % | 20.7 % | Finance | 23.8 % | 26.0 % |

## Simulated staff tables

300 tables of 1000 rows; 30 % of the salaries missing; 20 filled tables per multiple imputation. Truth: mean salary 64,561 NPR, standard deviation 25,977 NPR, correlation with experience 0.708, slope 2,199 NPR per year.

### MCAR

| Method | RMS error of the filled values / SD | Mean minus truth (NPR) | SD / truth | Correlation | Slope / truth | 95 % intervals containing the true mean |
|---|---|---|---|---|---|---|
| complete cases | - | +5 | 0.999 | 0.709 | 1.000 | 94 % |
| mean | 1.00 | +5 | 0.837 | 0.593 | 0.701 | 81 % |
| group mean | 0.92 | -25 | 0.866 | 0.573 | 0.701 | 86 % |
| nearest neighbours | 0.64 | -86 | 0.949 | 0.729 | 0.976 | 90 % |
| regression | 0.58 | -6 | 0.949 | 0.747 | 1.000 | 90 % |
| regression with noise | 0.82 | -17 | 1.000 | 0.709 | 1.002 | 89 % |
| multiple imputation | - | -12 | 1.000 | 0.708 | 1.000 | 94 % |

### MAR

| Method | RMS error of the filled values / SD | Mean minus truth (NPR) | SD / truth | Correlation | Slope / truth | 95 % intervals containing the true mean |
|---|---|---|---|---|---|---|
| complete cases | - | -6,873 | 0.875 | 0.594 | 1.001 | 0 % |
| mean | 1.33 | -6,873 | 0.732 | 0.364 | 0.376 | 0 % |
| group mean | 1.27 | -6,861 | 0.766 | 0.348 | 0.377 | 0 % |
| nearest neighbours | 0.70 | -1,452 | 0.892 | 0.682 | 0.859 | 47 % |
| regression | 0.58 | +1 | 0.949 | 0.747 | 1.002 | 90 % |
| regression with noise | 0.82 | +38 | 1.002 | 0.709 | 1.004 | 89 % |
| multiple imputation | - | +18 | 1.002 | 0.709 | 1.003 | 95 % |

### MNAR

| Method | RMS error of the filled values / SD | Mean minus truth (NPR) | SD / truth | Correlation | Slope / truth | 95 % intervals containing the true mean |
|---|---|---|---|---|---|---|
| complete cases | - | -8,897 | 0.830 | 0.616 | 0.855 | 0 % |
| mean | 1.45 | -8,897 | 0.694 | 0.435 | 0.426 | 0 % |
| group mean | 1.31 | -7,751 | 0.721 | 0.438 | 0.446 | 0 % |
| nearest neighbours | 0.82 | -4,229 | 0.831 | 0.689 | 0.809 | 0 % |
| regression | 0.70 | -3,381 | 0.867 | 0.734 | 0.898 | 2 % |
| regression with noise | 0.89 | -3,354 | 0.920 | 0.692 | 0.899 | 4 % |
| multiple imputation | - | -3,376 | 0.919 | 0.692 | 0.899 | 4 % |

## Area of the polygon

`data/new_shapefile.shp`: one polygon of 279 vertices, stored in EPSG:24047 (Indian 1975 / UTM zone 47N), at 84.9139 °E, 27.5856 °N, 14.1° west of the central meridian of that system, where its area factor is 1.0482. Area on the WGS 84 ellipsoid: 218,630 m².

| Coordinate system | Area (m²) | Difference from the area on the ellipsoid |
|---|---|---|
| as stored: Indian 1975 / UTM zone 47N | 229,175 | +4.82 % |
| WGS 84 / UTM zone 44N | 219,262 | +0.29 % |
| WGS 84 / UTM zone 45N | 218,684 | +0.02 % |
| transverse Mercator, central meridian 84 °E, scale 0.9999 | 218,630 | 0.00 % |
| Web Mercator | 279,381 | +27.79 % |
| equal-area cylindrical (EPSG:6933) | 218,629 | 0.00 % |

## Areas of countries

177 outlines (Natural Earth 1:110m), with vertices added so that none are more than 0.1° apart. Each area is divided by the area of the same outline on the ellipsoid.

- One square degree is between 2,046 and 12,305 km², depending on the country; 10,875 km² for Nepal.
- Web Mercator: between 1.007 (Gabon) and 16.4 (Greenland), median 1.21. Nepal 1.29, India 1.20, Brazil 1.06, Russia 4.88, Canada 5.20, Norway with Svalbard 9.34, Greenland 16.44. Left out, because it reaches beyond the latitudes that the projection shows: Antarctica.
- Equal-area cylindrical projection (EPSG:6933): within 0.0013 % for every country.
- UTM zone 45N used for every country: within 1 % for 6 (El Salvador, Guatemala, Belize, Bangladesh, Bhutan, Nepal); off by more than half for 98; not computable for 5 (Benin, Togo, Ghana, Côte d'Ivoire, Liberia); largest 83 (Ecuador). Nepal 1.002, India 1.023, China 1.095, Germany 1.598, Australia 1.883, Brazil 10.527.

## Transverse Mercator systems across Nepal (latitude 28 °N)

| System | Area error |
|---|---|
| UTM zone 44N | -0.06 % at 80.06 °E, +0.13 % at 84 °E, +1.17 % at 88.2 °E |
| UTM zone 45N | +1.08 % at 80.06 °E, +0.13 % at 84 °E, -0.05 % at 88.2 °E |
| UTM, zone chosen by longitude | -0.08 % to +0.13 % |
| 3° zones with central meridians 81, 84 and 87 °E and scale 0.9999 | -0.02 % to +0.03 % |
| Web Mercator | +28.8 % |
