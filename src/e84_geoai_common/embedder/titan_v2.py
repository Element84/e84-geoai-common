import logging
from typing import Literal, cast, get_args

import boto3
import botocore
import botocore.config
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from pydantic import BaseModel, ConfigDict, Field

from e84_geoai_common.embedder.embedder import (
    Embedder,
    EmbedderInferenceConfig,
    EmbedderInput,
    EmbedderResponse,
)

log = logging.getLogger(__name__)
TitanDimensions = Literal[1024, 512, 256]


# https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
class TitanV2InvokeRequest(BaseModel, frozen=True):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    input_text: str = Field(serialization_alias="inputText", alias="inputText")
    dimensions: TitanDimensions = Field(default=1024)
    normalize: bool = Field(default=True)
    embedding_types: list[Literal["float", "binary"]] = Field(
        default_factory=lambda: ["float"],
        serialization_alias="embeddingTypes",
        alias="embeddingTypes",
    )


class TitanV2Response(BaseModel, frozen=True):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    embedding: list[float] | None = None
    input_text_token_count: int = Field(
        serialization_alias="inputTextTokenCount", alias="inputTextTokenCount"
    )
    embeddings_by_type: dict[Literal["float", "binary"], list[float] | list[Literal[0, 1]]] = Field(
        serialization_alias="embeddingsByType", alias="embeddingsByType"
    )


class TitanV2(Embedder):
    client: BedrockRuntimeClient

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        client: BedrockRuntimeClient | None = None,
    ) -> None:
        """Initializes the embedder with the given model ID and client.

        Args:
            model_id: The Bedrock model ID to use for embedding.
            client: Optional pre-initialized Bedrock runtime boto3 client. Defaults to None.
        """
        self.model_id = model_id
        self.client = client or boto3.client(  # type: ignore[reportUnknownMemberType]
            "bedrock-runtime",
            config=botocore.config.Config(read_timeout=300),
        )

    def create_request(
        self,
        embedder_input: EmbedderInput,
        config: EmbedderInferenceConfig,
    ) -> TitanV2InvokeRequest:
        if config.dimensions not in get_args(TitanDimensions):
            raise ValueError(f"Dimensions must be one of {get_args(TitanDimensions)}")

        dimensions = cast("TitanDimensions", config.dimensions)
        return TitanV2InvokeRequest(
            inputText=embedder_input.input_text,
            dimensions=dimensions,
            normalize=config.normalize,
            embeddingTypes=config.embedding_types,
        )

    def embed(
        self,
        embedder_input: EmbedderInput,
        config: EmbedderInferenceConfig,
    ) -> EmbedderResponse:
        """Generates an embedding for the given input text."""
        request = self.create_request(embedder_input=embedder_input, config=config)
        response = self.invoke_model_with_request(request)
        return self.response_to_embedder_response(response)

    def invoke_model_with_request(self, request: TitanV2InvokeRequest) -> TitanV2Response:
        response_dict = self.client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=request.model_dump_json(by_alias=True),
        )
        response_body = response_dict["body"].read().decode("utf-8")
        response = TitanV2Response.model_validate_json(response_body)
        return response

    def response_to_embedder_response(self, response: TitanV2Response) -> EmbedderResponse:
        if response.embedding is None:
            raise ValueError("Binary embeddings are not supported in this method.")
        return EmbedderResponse(
            embedding=response.embedding,
            input_text_token_count=response.input_text_token_count,
        )
