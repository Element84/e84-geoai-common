from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EmbedderInferenceConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    normalize: bool = Field(default=True, description="Whether to normalize the embeddings.")
    dimensions: int = Field(default=1024, description="The dimensionality of the embeddings.")
    embedding_types: list[Literal["float", "binary"]] = Field(
        default_factory=lambda: ["float"], description="Types of embeddings to generate."
    )


class EmbedderInput(BaseModel, frozen=True):
    model_config = ConfigDict(strict=True, extra="forbid")

    input_text: str


class EmbedderResponse(BaseModel, frozen=True):
    model_config = ConfigDict(strict=True, extra="forbid")

    # FUTURE support binary embeddings and 2 embeddings at once
    embedding: list[float]
    input_text_token_count: int


class Embedder(ABC):
    """Abstract base class for all embedders."""

    model_id: str

    @abstractmethod
    def embed(
        self, embedder_input: EmbedderInput, config: EmbedderInferenceConfig
    ) -> EmbedderResponse:
        """Generates an embedding for the given input text.

        returns A list of floats or binary representing the embedding.
        """
