from math import cos, pi, sin

import pytest
from shapely import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    count_coordinates,
)
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import Polygon
from shapely.validation import explain_validity

from e84_geoai_common.geometry import (
    geometry_to_polygon,
    remove_extraneous_geoms,
    simplify_geometry,
)


def generate_circle(
    *, x: float = 0, y: float = 0, radius: float = 10, num_points: int = 10
) -> Polygon:
    angle = 2 * pi / num_points
    points = [
        (x + radius * cos(i * angle), y + radius * sin(i * angle)) for i in range(num_points - 1)
    ]
    return Polygon(points)


def generate_line_string(
    *,
    start_point: Point = Point(0, 0),  # noqa: B008
    end_point: Point = Point(10, 10),  # noqa: B008
    num_points: int = 10,
) -> LineString:
    """Generates a line string with points between start and end."""
    # Extract coordinates from the points
    x1, y1 = start_point.x, start_point.y
    x2, y2 = end_point.x, end_point.y

    # Calculate step size for interpolation
    x_step = (x2 - x1) / (num_points - 1) if num_points > 1 else 0
    y_step = (y2 - y1) / (num_points - 1) if num_points > 1 else 0

    # Generate the points along the line
    points = [(x1 + (i * x_step), y1 + (i * y_step)) for i in range(num_points)]
    return LineString(points)


def assert_remove_extraneous_geom_value_error(g: BaseGeometry, max_points: int) -> None:
    with pytest.raises(ValueError, match=r"Could not remove enough geometry parts"):
        remove_extraneous_geoms(g, max_points=max_points)


# A simple diagram showing the geometry for the following test.
#                A                     B              C
#    ┌─────────────────────┐       ┌──────────┐     ┌───┐
# 15 │                     │       │          │     │   │
#    │ ┌────────┐          │       │  B1      │     └───┘
#    │ │ A1     │          │       │  ┌──┐    │
#    │ │        │          │       │  └──┘    │
#    │ │        │          │       │          │
# 10 │ └────────┘          │       │   B2     │
#    │                     │       │  ┌───┐   │
#    │             A2      │       │  └───┘   │
#    │            ┌─┐      │       │          │
#    │            └─┘      │       │          │
# 5  │                     │       │   B3     │
#    └─────────────────────┘       │  ┌──┐    │
#                                  │  └──┘    │
#                                  │          │
# 1                                └──────────┘
#    5   10   15   20   25   30   35   40   45   50   55   60

a_exterior = [(4, 5), (29, 5), (29, 16), (4, 16), (4, 5)]
a_interior_1 = [(5, 10), (5, 15), (15, 15), (15, 10), (5, 10)]
a_interior_2 = [(15, 7), (15, 9), (20, 9), (20, 7), (15, 7)]
b_exterior = [(35, 1), (47, 1), (47, 16), (35, 16), (35, 1)]
b_interior_1 = [(37, 12), (37, 14), (43, 14), (43, 12), (37, 12)]
b_interior_2 = [(37, 7), (37, 9), (43, 9), (43, 7), (37, 7)]
b_interior_3 = [(37, 2), (37, 4), (41, 4), (41, 2), (37, 2)]
c_exterior = [(50, 13), (55, 13), (55, 15), (50, 15), (50, 13)]

abc_multipolygon = MultiPolygon(
    [
        Polygon(a_exterior, [a_interior_1, a_interior_2]),
        Polygon(
            b_exterior,
            [
                b_interior_1,
                b_interior_2,
                b_interior_3,
            ],
        ),
        Polygon(c_exterior),
    ]
)


def test_remove_extraneous_geoms_multipolygon():
    # Tuples of max points to the expected geometry from the abc multipolygon
    expected_geoms = [
        (5, MultiPolygon([Polygon(a_exterior)])),
        (10, MultiPolygon([Polygon(a_exterior), Polygon(b_exterior)])),
        (15, MultiPolygon([Polygon(a_exterior, [a_interior_1]), Polygon(b_exterior)])),
        (
            20,
            MultiPolygon(
                [
                    Polygon(a_exterior, [a_interior_1]),
                    Polygon(b_exterior, [b_interior_1]),
                ]
            ),
        ),
        (
            25,
            MultiPolygon(
                [
                    Polygon(a_exterior, [a_interior_1]),
                    Polygon(b_exterior, [b_interior_1, b_interior_2]),
                ]
            ),
        ),
        (
            30,
            MultiPolygon(
                [
                    Polygon(a_exterior, [a_interior_1, a_interior_2]),
                    Polygon(b_exterior, [b_interior_1, b_interior_2]),
                ]
            ),
        ),
        (
            35,
            MultiPolygon(
                [
                    Polygon(a_exterior, [a_interior_1, a_interior_2]),
                    Polygon(b_exterior, [b_interior_1, b_interior_2]),
                    Polygon(c_exterior),
                ]
            ),
        ),
        (
            40,
            MultiPolygon(
                [
                    Polygon(a_exterior, [a_interior_1, a_interior_2]),
                    Polygon(b_exterior, [b_interior_1, b_interior_2, b_interior_3]),
                    Polygon(c_exterior),
                ]
            ),
        ),
    ]
    for max_points, expected in expected_geoms:
        reduced = remove_extraneous_geoms(abc_multipolygon, max_points=max_points)
        assert reduced == expected
        assert reduced.is_valid, explain_validity(reduced)


def test_remove_extraneous_geometry_simple():
    # Point
    assert remove_extraneous_geoms(Point(0, 0), max_points=1) == Point(0, 0)
    assert_remove_extraneous_geom_value_error(Point(0, 0), max_points=0)

    # Line string
    linestring = generate_line_string(
        start_point=Point(0, 0), end_point=Point(10, 10), num_points=10
    )

    assert remove_extraneous_geoms(linestring, max_points=11) == linestring
    assert_remove_extraneous_geom_value_error(linestring, max_points=9)

    # Polygon
    polygon = generate_circle()
    assert remove_extraneous_geoms(polygon, max_points=11) == polygon
    assert_remove_extraneous_geom_value_error(polygon, max_points=9)


def test_remove_extraneous_geometry_misc():
    line10 = generate_line_string(end_point=Point(10, 10), num_points=10)
    line20 = generate_line_string(end_point=Point(20, 20), num_points=20)
    line5 = generate_line_string(end_point=Point(5, 5), num_points=5)

    # Multilinestring
    mls = MultiLineString([line10, line20, line5])  # total 35
    assert remove_extraneous_geoms(mls, max_points=40) == mls
    assert remove_extraneous_geoms(mls, max_points=30) == MultiLineString([line20, line10])
    assert remove_extraneous_geoms(mls, max_points=20) == line20
    # Removing the line20 but keeping the others would end up removing the larger area first
    # which isn't what we want.
    assert_remove_extraneous_geom_value_error(mls, max_points=19)

    # Geometry collection
    poly10 = generate_circle(radius=10, num_points=10)
    poly20 = generate_circle(radius=20, num_points=20)
    poly5 = generate_circle(radius=5, num_points=5)

    gc = GeometryCollection([poly10, poly20, poly5, line10, line20, line5])  # total 70
    assert remove_extraneous_geoms(gc, max_points=70) == gc
    assert remove_extraneous_geoms(gc, max_points=65) == GeometryCollection(
        [line20, line10, MultiPolygon([poly20, poly10, poly5])]
    )
    assert remove_extraneous_geoms(gc, max_points=60) == GeometryCollection(
        [line20, line10, MultiPolygon([poly20, poly10])]
    )
    assert remove_extraneous_geoms(gc, max_points=40) == GeometryCollection(
        [line20, MultiPolygon([poly20])]
    )
    assert remove_extraneous_geoms(gc, max_points=30) == MultiPolygon([poly20])
    assert_remove_extraneous_geom_value_error(gc, max_points=19)


def test_simplify_geometry():
    point = Point(1, 1)
    assert simplify_geometry(point) == point

    polygon = generate_circle(num_points=200)
    # Simplifies to the same thing if already less than the set number of
    # points
    assert simplify_geometry(polygon, 300) == polygon

    assert count_coordinates(simplify_geometry(polygon, 199)) <= 199


# A simple ascii art drawing for another set of sample polygons
# 12   ************     ******
#      *  A       *     * B  *
# 10   *     ******     ******        *
#      *     *                       /
#      *     *                      / E
# 7    *     ******       D        *
#      *          *       *
# 5    *     ******
#      *     *
# 3    *     ******     ******
#      *          *     * C  *
# 1    ************     ******
#      4     9    11    13  15    17  19


def test_geometry_to_polygon():
    a_polygon = Polygon(
        [
            (4, 1),
            (11, 1),
            (11, 3),
            (9, 3),
            (9, 5),
            (11, 5),
            (11, 7),
            (9, 7),
            (9, 10),
            (11, 10),
            (11, 12),
            (4, 12),
            (4, 1),
        ]
    )

    b_polygon = Polygon([(13, 10), (15, 10), (15, 12), (13, 12), (13, 10)])
    c_polygon = Polygon([(13, 1), (15, 1), (15, 3), (13, 3), (13, 1)])
    c_bottom_line = LineString([(13, 1), (15, 1)])

    d_point = Point(14, 6)
    e_line = LineString([(19, 10), (17, 7)])
    e1_point = Point(19, 10)
    e2_point = Point(17, 7)

    ab_multipolygon = MultiPolygon([a_polygon, b_polygon])
    ab_convex_hull = Polygon([(4, 1), (11, 1), (15, 10), (15, 12), (4, 12), (4, 1)])

    abcde_collection = GeometryCollection([a_polygon, b_polygon, c_polygon, d_point, e_line])
    abcde_convex_hull = Polygon([(4, 1), (15, 1), (19, 10), (15, 12), (4, 12), (4, 1)])

    # A single polygon is returned
    assert geometry_to_polygon(a_polygon) == a_polygon

    # Multipolygon convex hull is used
    assert geometry_to_polygon(ab_multipolygon).equals(ab_convex_hull)

    # Point adds a buffer
    assert geometry_to_polygon(d_point) == d_point.buffer(1)
    # Line string adds a buffer
    assert geometry_to_polygon(e_line) == e_line.buffer(1)

    # Geometry collection is convex hull
    assert geometry_to_polygon(abcde_collection).equals(abcde_convex_hull)

    # Multipoints
    de_multipoint = MultiPoint([d_point, e1_point, e2_point])
    e_multipoint = MultiPoint([e1_point, e2_point])

    # Convex hull turns into a polygon
    assert geometry_to_polygon(de_multipoint).equals(Polygon([d_point, e1_point, e2_point]))
    # Convex hull turns into a line string
    assert geometry_to_polygon(e_multipoint).equals(LineString([e1_point, e2_point]).buffer(1))

    # MultiLinestring
    ce_multiline = MultiLineString([c_bottom_line, e_line])
    assert geometry_to_polygon(ce_multiline).equals(Polygon([(13, 1), (15, 1), (19, 10), (13, 1)]))
