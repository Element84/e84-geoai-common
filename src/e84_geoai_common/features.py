"""TODO"""

from typing import Annotated, Any, Generic, Literal, TypeVar, cast
from pydantic import (
    BaseModel,
    ConfigDict,
    SkipValidation,
    field_validator,
    field_serializer,
)
from shapely.geometry.base import BaseGeometry

from e84_geoai_common.geometry import geometry_from_geojson_dict


T = TypeVar("T")


class Feature(BaseModel, Generic[T]):
    """TODO"""

    # Extra fields are allowed to be open like GeoJSON is.
    model_config = ConfigDict(strict=True, frozen=True, arbitrary_types_allowed=True)

    type: Literal["Feature"] = "Feature"
    geometry: Annotated[BaseGeometry, SkipValidation]
    properties: T

    @field_validator("geometry", mode="before")
    @classmethod
    def parse_shapely_geometry(cls, d: Any) -> BaseGeometry:
        if isinstance(d, dict):
            return geometry_from_geojson_dict(cast(dict[str, Any], d))
        elif isinstance(d, BaseGeometry):
            return d
        else:
            raise Exception(
                "geometry must be a geojson feature dictionary or a shapely geometry"
            )

    @field_serializer("geometry")
    def shapely_geometry_to_json(self, g: BaseGeometry) -> dict[str, Any]:
        return g.__geo_interface__


class FeatureCollection(BaseModel, Generic[T]):
    """TODO"""

    # Extra fields are allowed to be open like GeoJSON is.
    model_config = ConfigDict(strict=True, frozen=True)
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[Feature[T]]
