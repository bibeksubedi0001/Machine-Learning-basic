"""Areas on the ellipsoid and in map projections."""
from __future__ import annotations

import numpy as np
from pyproj import CRS, Geod, Proj, Transformer
from shapely.ops import transform as apply

WGS84 = Geod(ellps="WGS84")
ECCENTRICITY_SQUARED = WGS84.es


def geodesic_area(geometry):
    """Area in m² on the WGS 84 ellipsoid of a polygon given in longitude and latitude."""
    return abs(WGS84.geometry_area_perimeter(geometry)[0])


def to_lonlat(geometry, crs):
    """A geometry of the coordinate system `crs` in WGS 84 longitude and latitude."""
    return apply(Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform, geometry)


def projected_area(geometry, crs):
    """Plane area of a longitude-latitude polygon after projecting it to `crs`, in the units of `crs` squared."""
    return apply(Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform, geometry).area


def web_mercator_area_factor(latitude):
    """Plane area in Web Mercator (EPSG:3857) divided by the area on the ellipsoid, at a latitude in degrees.

    Web Mercator applies the formulas of the sphere to coordinates on the ellipsoid. Its scale is
    sqrt(1 - e² sin² φ) / cos φ along the parallel and (1 - e² sin² φ)^1.5 / ((1 - e²) cos φ) along the
    meridian; the factor is their product, close to 1 / cos² φ.
    """
    phi = np.radians(latitude)
    w = 1.0 - ECCENTRICITY_SQUARED * np.sin(phi) ** 2
    return w**2 / ((1.0 - ECCENTRICITY_SQUARED) * np.cos(phi) ** 2)


def area_factor(crs, longitude, latitude):
    """Plane area divided by ellipsoid area at a point, for a projection that PROJ gives scale factors for.

    Right for the transverse Mercator systems used here; not for Web Mercator, where PROJ reports the
    factor of the sphere (use `web_mercator_area_factor`).
    """
    return Proj(CRS.from_user_input(crs)).get_factors(longitude, latitude).areal_scale


def transverse_mercator(central_meridian, scale=0.9996):
    """A transverse Mercator system on the WGS 84 ellipsoid: UTM with scale 0.9996, Nepal's MUTM with 0.9999."""
    return CRS.from_proj4(f"+proj=tmerc +lat_0=0 +lon_0={central_meridian} +k={scale} +x_0=500000 +y_0=0 +ellps=WGS84 +units=m")
