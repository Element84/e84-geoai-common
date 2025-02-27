from math import cos, pi, sin

from shapely import GeometryCollection, LineString, Point
from shapely.geometry.polygon import Polygon

from e84_geoai_common.geometry import geometry_point_count, simplify_geometry


def generate_circle(
    *, x: float = 0, y: float = 0, radius: float = 10, num_points: int = 10
) -> Polygon:
    angle = 2 * pi / num_points
    points = [(x + radius * cos(i * angle), y + radius * sin(i * angle)) for i in range(num_points)]
    return Polygon(points)


def test_geometry_point_count():
    # Point
    assert geometry_point_count(Point(1, 1)) == 1

    # Polygon with exterior ring
    assert geometry_point_count(generate_circle(num_points=10)) == 10

    # Polygon with hole
    exterior = generate_circle(radius=20, num_points=10)
    interior = generate_circle(radius=5, num_points=7)
    polygon_with_hole = exterior.difference(interior)
    assert geometry_point_count(polygon_with_hole) == 17

    # MultiPolygon
    other_poly = generate_circle(x=40, y=40, num_points=12)
    multipolygon = polygon_with_hole.union(other_poly)
    assert geometry_point_count(multipolygon) == 17 + 12

    # Linear Ring
    assert geometry_point_count(exterior.exterior) == 10

    # MultiPoint
    multipoint = Point(1, 1).union(Point(2, 2)).union(Point(3, 3))
    assert geometry_point_count(multipoint) == 3

    # LineString
    line1 = LineString([(0, 0), (1, 1), (2, 2)])
    assert geometry_point_count(line1) == 3

    line2 = LineString([(0, 1), (1, 2), (2, 3), (3, 3)])
    assert geometry_point_count(line2) == 4

    # MultiLineString
    multiline = line1.union(line2)
    assert geometry_point_count(multiline) == 7

    # Geometry Collection
    collection = GeometryCollection([multipolygon, multipoint, multiline])
    assert geometry_point_count(collection) == 17 + 12 + 3 + 7


def test_simplify_geometry():
    point = Point(1, 1)
    assert simplify_geometry(point) == point

    polygon = generate_circle(num_points=200)
    # Simplifies to the same thing if already less than the set number of
    # points
    assert simplify_geometry(polygon, 300) == polygon

    assert geometry_point_count(simplify_geometry(polygon, 199)) <= 199
