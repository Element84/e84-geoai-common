"""Helpers for dealing with geometry."""

import json
import math
from typing import Any

import shapely
import shapely.geometry
from shapely import GeometryCollection
from shapely.geometry.base import BaseGeometry

from e84_geoai_common.util import timed_function


def geometry_from_wkt(wkt: str) -> BaseGeometry:
    """Create shapely geometry from Well-Known Text (WKT) string."""
    return shapely.from_wkt(wkt)  # type: ignore[reportUnknownVariableType]


def geometry_from_geojson_dict(geom: dict[str, Any]) -> BaseGeometry:
    """Create shapely geometry from GeoJSON dict.

    Example:
        Sample usage of the function:

        >>> geometry_dict = {
        ...    "type": "Point",
        ...    "coordinates": [100.0, 0.0]
        ... }
        >>> shapely_geometry = geometry_from_geojson_dict(geometry_dict)
        >>> print(shapely_geometry)
        POINT (100 0)
    """
    return shapely.geometry.shape(geom)


def geometry_from_geojson(geojson: str) -> BaseGeometry:
    """Create shapely geometry from GeoJSON string."""
    return geometry_from_geojson_dict(json.loads(geojson))


def geometry_to_geojson(geom: BaseGeometry) -> str:
    """Convert shapely geometry to GeoJSON string.

    Example:
        Sample usage of the function:

        >>> from shapely.geometry import Point
        >>> point = Point(0, 0)
        >>> geojson_string = geometry_to_geojson(point)
        >>> print(geojson_string)
        '{"type": "Point", "coordinates": [0.0, 0.0]}'
    """
    return json.dumps(geom.__geo_interface__)


def geometry_point_count(geom: BaseGeometry) -> int:  # noqa: PLR0911
    """Count number of points in exterior and interior (if any) of geometry."""
    if isinstance(geom, shapely.geometry.Point):
        return 1
    if isinstance(geom, shapely.geometry.MultiPoint):
        return sum(geometry_point_count(g) for g in geom.geoms)
    if isinstance(geom, shapely.geometry.Polygon):
        exterior_count = geometry_point_count(geom.exterior)
        interior_count = sum(geometry_point_count(interior) for interior in geom.interiors)
        return exterior_count + interior_count
    if isinstance(geom, shapely.geometry.MultiPolygon):
        return sum(geometry_point_count(g) for g in geom.geoms)
    if isinstance(geom, shapely.geometry.LinearRing):
        return len(geom.coords) - 1
    if isinstance(geom, shapely.geometry.LineString):
        return len(geom.coords)
    if isinstance(geom, shapely.geometry.MultiLineString):
        return sum(len(line.coords) for line in geom.geoms)
    if isinstance(geom, shapely.geometry.GeometryCollection):
        return sum(geometry_point_count(g) for g in geom.geoms)  # type: ignore[reportUnknownVariableType]
    msg = f"Unsupported geometry type: {type(geom).__name__}"
    raise TypeError(msg)


@timed_function
def simplify_geometry(geom: BaseGeometry, max_points: int = 3_000) -> BaseGeometry:
    """Simplify geometry.

    Simplifies a shapely geometry object by reducing the number of points in
    the geometry while preserving its overall shape.

    Args:
        geom (BaseGeometry): The shapely geometry object to be simplified.
        max_points (int): The maximum number of points allowed in the
            simplified geometry.

    Raises:
        ValueError: If geometry cannot be simplified to under max_points
            points.
    """
    num_points = geometry_point_count(geom)
    if num_points < max_points:
        return geom
    # Repeatedly try different tolerances to simplify it just until it reaches
    # below the maximum number of points.
    for power in range(-7, 0):
        tolerance: float = pow(10, power)
        simplified = geom.simplify(tolerance)
        if geometry_point_count(simplified) < max_points:
            return simplified
    msg = "Unable to simplify the geometry enough to get it under the maximum number of points"
    raise ValueError(msg)


def BoundingBox(  # noqa: N802
    *, west: float, south: float, east: float, north: float
) -> BaseGeometry:
    """Construct a bounding box geometry using the provided coordinates.

    Args:
        west (float): The western longitude of the bounding box.
        south (float): The southern latitude of the bounding box.
        east (float): The eastern longitude of the bounding box.
        north (float): The northern latitude of the bounding box.

    Returns:
        BaseGeometry: A shapely geometry object representing the bounding box.

    Example:
        Sample usage of the function:

        >>> box = BoundingBox(west=-74.1, south=40.5, east=-73.9, north=40.8)
        >>> print(box)
        POLYGON ((-74.1 40.5, -73.9 40.5, -73.9 40.8, -74.1 40.8, -74.1 40.5))
    """
    return shapely.geometry.box(west, south, east, north)


def between(g1: BaseGeometry, g2: BaseGeometry) -> BaseGeometry:
    """Return the geometry between the two geometries."""
    coll = GeometryCollection([g1, g2])
    return coll.convex_hull - g1.convex_hull - g2.convex_hull


def degrees_to_radians(deg: float) -> float:
    """Convert degrees to radians."""
    return deg * (math.pi / 180.0)


def add_buffer(g: BaseGeometry, distance_km: float) -> BaseGeometry:
    """Add a buffer around the input geometry.

    This function calculates the latitude and longitude distances based on the
    input distance in kilometers. The longitude distance is adjusted based on
    the average latitude of the input geometry. A buffer of the calculated
    average distance between longitude and latitude is added around the input
    geometry.

    Args:
        g (BaseGeometry): The input geometry for which the buffer is to be
            added.
        distance_km (float): The distance in kilometers for the buffer.

    Returns:
        BaseGeometry: A new geometry object with the buffer added around the
            input geometry.

    Note:
        - This function assumes a basic circle buffer and may not work
        correctly near the poles.
        - For accurate distance calculations, consider using a more
        sophisticated approach.

    Example:
        Sample usage of the function:

        >>> from shapely.geometry import Point
        >>> point = Point(0, 0)
        >>> buffered_point = add_buffer(point, 5.0)
        >>> print(buffered_point)
    """
    lat_distance = distance_km / 111.32

    # longitude distance varies with latitude
    avg_lat = g.centroid.y
    avg_lat_rad = degrees_to_radians(avg_lat)
    lon_distance = distance_km / (math.cos(avg_lat_rad) * 111.32)

    # Since we're creating a basic circle around the geometry we'll do
    # something dumb here and just average the longitude and latitude distance.
    # This will fall apart at the poles but works for our current use cases.

    return g.buffer((lon_distance + lat_distance) / 2.0)
