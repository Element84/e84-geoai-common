"""TODO"""

import json
import math
from typing import Any, Protocol, Sequence, runtime_checkable
import shapely
import shapely.geometry

from e84_geoai_common.util import timed_function


@runtime_checkable
class Geometry(Protocol):
    """TODO"""

    @property
    def convex_hull(self) -> "Geometry": ...

    @property
    def centroid(self) -> "Point": ...

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Returns min_lon, min_lat, max_lon, max_lat"""
        ...

    def simplify(
        self, tolerance: float, preserve_topology: bool = True
    ) -> "Geometry": ...

    def __len__(self) -> int: ...

    @property
    def __geo_interface__(self) -> dict[str, Any]:
        """Returns geojson"""
        ...

    @property
    def is_empty(self) -> bool: ...

    def buffer(self, distance: float) -> "Geometry": ...

    def __and__(self, other: "Geometry") -> "Geometry": ...
    def intersection(self, other: "Geometry") -> "Geometry": ...

    def __or__(self, other: "Geometry") -> "Geometry": ...
    def union(self, other: "Geometry") -> "Geometry": ...

    def __sub__(self, other: "Geometry") -> "Geometry": ...
    def difference(self, other: "Geometry") -> "Geometry": ...

    def __xor__(self, other: "Geometry") -> "Geometry": ...
    def symmetric_difference(self, other: "Geometry") -> "Geometry": ...


class Point(Geometry):
    """TODO"""

    @property
    def x(self) -> int: ...

    @property
    def y(self) -> int: ...


def GeometryCollection(geoms: Sequence[Geometry]) -> Geometry:
    """TODO"""
    return shapely.GeometryCollection(geoms)  # type: ignore


def geometry_from_wkt(wkt: str) -> Geometry:
    """TODO"""
    return shapely.from_wkt(wkt)  # type: ignore


def geometry_from_geojson_dict(geom: dict[str, Any]) -> Geometry:
    """TODO"""
    return shapely.geometry.shape(geom)  # type: ignore


def geometry_from_geojson(geojson: str) -> Geometry:
    """TODO"""
    return geometry_from_geojson_dict(json.loads(geojson))


def geometry_to_geojson(geom: Geometry) -> str:
    """TODO"""
    return json.dumps(geom.__geo_interface__)


def geometry_point_count(geom: Geometry) -> int:
    """Returns the number of points in both exterior and interior (if any) in a Geometry."""
    if isinstance(geom, shapely.geometry.Point):
        return 1
    elif isinstance(geom, shapely.geometry.MultiPoint):
        return len(geom)
    elif isinstance(geom, shapely.geometry.Polygon):
        exterior_count = len(geom.exterior.coords)  # type: ignore
        interior_count = sum(len(interior.coords) for interior in geom.interiors)  # type: ignore
        return exterior_count + interior_count
    elif isinstance(geom, shapely.geometry.MultiPolygon):
        return sum(
            len(p.exterior.coords)  # type: ignore
            + sum(len(interior.coords) for interior in p.interiors)  # type: ignore
            for p in geom.geoms  # type: ignore
        )
    elif isinstance(geom, shapely.geometry.LineString):
        return len(geom.coords)
    elif isinstance(geom, shapely.geometry.MultiLineString):
        return sum(len(line.coords) for line in geom.geoms)  # type: ignore
    elif isinstance(geom, shapely.geometry.GeometryCollection):
        return sum(geometry_point_count(g) for g in geom.geoms)  # type: ignore
    else:
        raise TypeError(f"Unsupported geometry type: {type(geom).__name__}")


@timed_function
def simplify_geometry(geom: Geometry, max_points: int = 3_000) -> Geometry:
    """TODO"""
    num_points = geometry_point_count(geom)
    if num_points < max_points:
        return geom
    for power in range(-7, 0):
        tolerance = pow(10, power)
        simplified = geom.simplify(tolerance)
        if geometry_point_count(simplified) < max_points:
            return simplified
    raise Exception(
        "Unable to simplify the geometry enough to get it under the maximum number of points"
    )


def BoundingBox(*, west: float, south: float, east: float, north: float) -> Geometry:
    """TODO"""
    return shapely.geometry.box(west, south, east, north)  # type: ignore


def between(g1: Geometry, g2: Geometry) -> Geometry:
    """Returns the geometry between the two geometries"""
    coll = GeometryCollection([g1, g2])
    return coll.convex_hull - g1.convex_hull - g2.convex_hull


def degrees_to_radians(deg: float) -> float:
    """TODO"""
    return deg * (math.pi / 180.0)


def add_buffer(g: Geometry, distance_km: float) -> Geometry:
    """TODO"""
    lat_distance = distance_km / 111.32

    # longitude distance varies with latitude
    avg_lat = g.centroid.y
    avg_lat_rad = degrees_to_radians(avg_lat)
    lon_distance = distance_km / (math.cos(avg_lat_rad) * 111.32)

    # Since we're creating a basic circle around the geometry we'll do something dumb here
    # and just average the longitude and latitude distance. This will fall apart at the poles
    # but works for our demo use cases.

    return g.buffer((lon_distance + lat_distance) / 2.0)
