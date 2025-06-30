"""Pydantic models for GeoJSON features."""

from typing import Annotated, Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    SkipValidation,
    field_serializer,
    field_validator,
)
from shapely.geometry.base import BaseGeometry

from e84_geoai_common.geometry import geometry_from_geojson_dict


class Feature[T](BaseModel):
    """Represents a feature object as defined in the GeoJSON format.

    Attributes:
        type: Literal["Feature"] - specifies the type of the GeoJSON object as
            'Feature'.
        geometry: Annotated[BaseGeometry, SkipValidation] - represents the
            geometry of the feature.
        properties: T - generic type representing the properties associated
            with the feature.
    """

    model_config = ConfigDict(strict=True, frozen=True, arbitrary_types_allowed=True)

    type: Literal["Feature"] = "Feature"
    geometry: Annotated[BaseGeometry, SkipValidation]
    properties: T

    @field_validator("geometry", mode="before")
    @classmethod
    def _parse_shapely_geometry(cls, d: Any) -> BaseGeometry:  # noqa: ANN401
        if isinstance(d, dict):
            return geometry_from_geojson_dict(cast("dict[str, Any]", d))
        if isinstance(d, BaseGeometry):
            return d
        msg = "geometry must be a geojson feature dictionary or a shapely geometry."
        raise TypeError(msg)

    @field_serializer("geometry")
    def _shapely_geometry_to_json(self, g: BaseGeometry) -> dict[str, Any]:
        return g.__geo_interface__


class FeatureCollection[T](BaseModel):
    """Represents a collection of feature objects in the GeoJSON format.

    Attributes:
        model_config: ConfigDict - configuration settings for the model.
        type: Literal["FeatureCollection"] - specifies the type of the GeoJSON
            object as 'FeatureCollection'.
        features: list[Feature[T]] - a list of features included in the
            collection.
    """

    # Extra fields are allowed to be open like GeoJSON is.
    model_config = ConfigDict(strict=True, frozen=True)
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[Feature[T]]
