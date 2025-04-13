from math import cos, pi, sin

from shapely import (
    MultiPolygon,
    Point,
    count_coordinates,
)
from shapely.geometry.polygon import Polygon
from shapely.validation import explain_validity

from e84_geoai_common.geometry import (
    remove_extraneous_geom,
    simplify_geometry,
)

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


def test_remove_extraneous_geoms():
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
        reduced = remove_extraneous_geom(abc_multipolygon, max_points=max_points)
        assert reduced == expected
        assert reduced.is_valid, explain_validity(reduced)


def generate_circle(
    *, x: float = 0, y: float = 0, radius: float = 10, num_points: int = 10
) -> Polygon:
    angle = 2 * pi / num_points
    points = [(x + radius * cos(i * angle), y + radius * sin(i * angle)) for i in range(num_points)]
    return Polygon(points)


def test_simplify_geometry():
    point = Point(1, 1)
    assert simplify_geometry(point) == point

    polygon = generate_circle(num_points=200)
    # Simplifies to the same thing if already less than the set number of
    # points
    assert simplify_geometry(polygon, 300) == polygon

    assert count_coordinates(simplify_geometry(polygon, 199)) <= 199
