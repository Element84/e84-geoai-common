"""Pydantic models for GeoJSON features"""

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
    """
    Represents a feature object as defined in the GeoJSON format.

    Attributes:
        type: Literal["Feature"] - specifies the type of the GeoJSON object as 'Feature'.
        geometry: Annotated[BaseGeometry, SkipValidation] - represents the geometry of the feature.
        properties: T - generic type representing the properties associated with the feature.
    """

    model_config = ConfigDict(strict=True, frozen=True, arbitrary_types_allowed=True)

    type: Literal["Feature"] = "Feature"
    geometry: Annotated[BaseGeometry, SkipValidation]
    properties: T

    @field_validator("geometry", mode="before")
    @classmethod
    def _parse_shapely_geometry(cls, d: Any) -> BaseGeometry:
        if isinstance(d, dict):
            return geometry_from_geojson_dict(cast(dict[str, Any], d))
        elif isinstance(d, BaseGeometry):
            return d
        else:
            raise Exception(
                "geometry must be a geojson feature dictionary or a shapely geometry"
            )

    @field_serializer("geometry")
    def _shapely_geometry_to_json(self, g: BaseGeometry) -> dict[str, Any]:
        return g.__geo_interface__


class FeatureCollection(BaseModel, Generic[T]):
    """
    Represents a collection of feature objects defined in the GeoJSON format.

    Attributes:
        model_config: ConfigDict - configuration settings for the model.
        type: Literal["FeatureCollection"] - specifies the type of the GeoJSON object as 'FeatureCollection'.
        features: list[Feature[T]] - a list of features included in the collection.
    """

    # Extra fields are allowed to be open like GeoJSON is.
    model_config = ConfigDict(strict=True, frozen=True)
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[Feature[T]]
