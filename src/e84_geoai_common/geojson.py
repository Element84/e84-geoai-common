"""Pydantic models for GeoJSON features"""

from typing import Annotated, Generic, Literal, TypeVar
from pydantic import (
    BaseModel,
    ConfigDict,
    SkipValidation,
)

from e84_geoai_common.geometry import GeometryWithSerialization


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
    geometry: Annotated[GeometryWithSerialization, SkipValidation]
    properties: T


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
