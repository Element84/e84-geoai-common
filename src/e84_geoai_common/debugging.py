import folium  # type: ignore
from shapely import GeometryCollection
from shapely.geometry.base import BaseGeometry

from e84_geoai_common.util import timed_function

SEARCH_AREA_STYLE = {"fillColor": "transparent", "color": "#FF0000", "weight": 3}


@timed_function
def display_geometry(
    geoms: list[BaseGeometry],
    *,
    selected_geometry: BaseGeometry | None = None,
    search_area: BaseGeometry | None = None,
) -> folium.Map:
    """
    Display the provided geometries on a folium Map with optional highlighting of a selected geometry or search area.

    Parameters:
    - geoms: List of shapely BaseGeometry objects to be displayed on the map.
    - selected_geometry: Optional BaseGeometry object to highlight on the map.
    - search_area: Optional BaseGeometry object to display as a search area on the map.

    Returns:
    Folium Map object displaying the provided geometries with optional highlighting.

    If a selected_geometry is provided, it will be highlighted on the map. If a search_area is provided,
    it will also be displayed on the map. If neither is provided, the center of the bounding box of all
    geometries will be calculated and used as the center of the map.

    The map will be fitted to display all provided geometries with the calculated center point.

    Note: The SEARCH_AREA_STYLE dictionary defines the style for the search area displayed on the map.
    """
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
