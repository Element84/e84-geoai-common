"""TODO"""

import folium  # type: ignore

from e84_geoai_common.geometry import Geometry, GeometryCollection
from e84_geoai_common.util import timed_function

SEARCH_AREA_STYLE = {"fillColor": "transparent", "color": "#FF0000", "weight": 3}


@timed_function
def display_geometry(
    geoms: list[Geometry],
    *,
    selected_geometry: Geometry | None = None,
    search_area: Geometry | None = None,
) -> folium.Map:
    """TODO"""

    if selected_geometry:
        min_lon, min_lat, max_lon, max_lat = selected_geometry.bounds
        point = selected_geometry.centroid
    elif search_area:
        min_lon, min_lat, max_lon, max_lat = search_area.bounds
        point = search_area.centroid
    else:
        # Calculate center of the bounding box
        coll = GeometryCollection(geoms)
        hull = coll.convex_hull
        point = hull.centroid
        min_lon, min_lat, max_lon, max_lat = hull.bounds

    sw = [min_lat, min_lon]
    ne = [max_lat, max_lon]

    # Create a map centered at the calculated center
    m = folium.Map(
        location=[point.y, point.x],
    )
    m.fit_bounds([sw, ne])

    for geom in geoms:
        g = folium.GeoJson(geom.__geo_interface__)
        g.add_to(m)  # type: ignore

    if search_area:
        g = folium.GeoJson(
            search_area.__geo_interface__,
            style_function=lambda _x: SEARCH_AREA_STYLE,  # type: ignore
        )
        g.add_to(m)  # type: ignore

    return m
