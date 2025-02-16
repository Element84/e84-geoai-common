import logging
from collections.abc import Sequence
from typing import Literal, Self

import boto3
import botocore.exceptions
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from pydantic import BaseModel, ConfigDict, Field

from e84_geoai_common.llm.core.llm import (
    LLM,
    Base64ImageContent,
    LLMInferenceConfig,
    LLMMessage,
    TextContent,
)
from e84_geoai_common.util import timed_function

log = logging.getLogger(__name__)

# https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html#model-ids-arns
NOVA_BEDROCK_MODEL_IDS = {
    "Nova Canvas": "amazon.nova-canvas-v1:0",
    "Nova Lite": "amazon.nova-lite-v1:0",
    "Nova Micro": "amazon.nova-micro-v1:0",
    "Nova Pro": "amazon.nova-pro-v1:0",
    "Nova Reel": "amazon.nova-reel-v1:0",
}


class NovaTextContent(BaseModel):
    """Nova text context model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    text: str


class NovaImageSource(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    bytes: str


class NovaImageContent(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    format: Literal["jpeg", "png", "gif", "webp"]
    source: NovaImageSource

    @classmethod
    def from_b64_image_content(cls, image: Base64ImageContent) -> Self:
        img_format = image.media_type.split("/")[-1]
        return cls(
            format=img_format,  # type: ignore[reportArgumentType]
            source=NovaImageSource(bytes=image.data),
        )

    def to_b64_image_content(self) -> Base64ImageContent:
        return Base64ImageContent(
            format=f"image/{self.format}",  # type: ignore[reportArgumentType]
            data=self.source.bytes,
        )


class NovaMessage(BaseModel):
    """Nova message base model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    role: Literal["assistant", "user"] = "user"
    content: Sequence[NovaTextContent | NovaImageContent]

    @classmethod
    def from_llm_message(cls, msg: LLMMessage) -> Self:
        def _handle_content(
            subcontent: TextContent | Base64ImageContent,
        ) -> NovaTextContent | NovaImageContent:
            if isinstance(subcontent, TextContent):
                return NovaTextContent(text=subcontent.text)
            return NovaImageContent.from_b64_image_content(subcontent)

        if isinstance(msg.content, str):
            content = [NovaTextContent(text=msg.content)]
        else:
            content = [_handle_content(subcontent) for subcontent in msg.content]
        return cls(role=msg.role, content=content)

    def to_llm_message(self, inference_cfg: LLMInferenceConfig) -> LLMMessage:
        def _to_llm_content(
            index: int,
            c: NovaTextContent | NovaImageContent,
        ) -> TextContent | Base64ImageContent:
            if isinstance(c, NovaTextContent):
                content = c.text
                if index == 0 and inference_cfg.response_prefix:
                    content = inference_cfg.response_prefix + content
                return TextContent(text=content)

            return c.to_b64_image_content()

        if len(self.content) == 1 and isinstance(self.content[0], NovaTextContent):
            content = self.content[0].text
            if inference_cfg.json_mode:
                # In JSON mode we need to remove the JSON stop sequence
                content = content.removesuffix("```")
            elif inference_cfg.response_prefix:
                content = inference_cfg.response_prefix + content
        else:
            content = [_to_llm_content(index, c) for index, c in enumerate(self.content)]

        return LLMMessage(role=self.role, content=content)


class NovaInferenceConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    max_new_tokens: int = Field(default=1000, description="Maximum number of output tokens")
    temperature: float = Field(
        default=0,
        description="Temperature control for randomness. Closer to zero = more deterministic.",
    )
    top_p: float | None = Field(default=None, description="Top P for nucleus sampling")
    top_k: int | None = Field(default=None, description="Top K for sampling")
    stop_sequences: list[str] | None = Field(default=None, serialization_alias="stopSequences")


class NovaInvokeLLMRequest(BaseModel):
    """Represents a request to invoke Nova and get a response back."""

    model_config = ConfigDict(strict=True, extra="forbid")

    system: list[NovaTextContent] | None = Field(default=None)

    messages: list[NovaMessage] = Field(default_factory=list)

    inference_config: NovaInferenceConfig | None = Field(
        default=None, serialization_alias="inferenceConfig"
    )


##################################################################
# Response Classes


class NovaResponseOutput(BaseModel):
    """Contains the output of the nova response."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    message: NovaMessage


class NovaUsageInfo(BaseModel):
    """Nova usage-info model."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    input_tokens: int = Field(alias="inputTokens")
    output_tokens: int = Field(alias="outputTokens")
    total_tokens: int = Field(alias="totalTokens")
    cache_read_input_token_count: int = Field(alias="cacheReadInputTokenCount")
    cache_write_input_token_count: int = Field(alias="cacheWriteInputTokenCount")


class NovaResponse(BaseModel):
    """Represents the response from a nova model request."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    output: NovaResponseOutput
    stop_reason: Literal["end_turn", "max_tokens"] = Field(alias="stopReason")
    usage: NovaUsageInfo


class BedrockNovaLLM(LLM):
    """Implements the LLM class for Bedrock Nova."""

    client: BedrockRuntimeClient

    def __init__(
        self,
        model_id: str = NOVA_BEDROCK_MODEL_IDS["Nova Pro"],
        client: BedrockRuntimeClient | None = None,
    ) -> None:
        """Initialize.

        Args:
            model_id: Model ID. Defaults to the model ID for Nova Pro.
            client: Optional pre-initialized boto3 client. Defaults to None.
        """
        self.model_id = model_id
        self.client = client or boto3.client("bedrock-runtime")  # type: ignore[reportUnknownMemberType]

    def _create_request(
        self, messages: Sequence[LLMMessage], config: LLMInferenceConfig
    ) -> NovaInvokeLLMRequest:
        system = [NovaTextContent(text=config.system_prompt)] if config.system_prompt else None
        stop_sequences = None
        if config.json_mode:
            # https://docs.aws.amazon.com/nova/latest/userguide/prompting-structured-output.html
            prefix = "```json"
            messages = [*messages, LLMMessage(role="assistant", content=prefix)]
            stop_sequences = ["```"]
        elif config.response_prefix:
            messages = [*messages, LLMMessage(role="assistant", content=config.response_prefix)]
        return NovaInvokeLLMRequest(
            system=system,
            inference_config=NovaInferenceConfig(
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                stop_sequences=stop_sequences,
            ),
            messages=[NovaMessage.from_llm_message(msg) for msg in messages],
        )

    @timed_function
    def prompt(
        self,
        messages: Sequence[LLMMessage],
        inference_cfg: LLMInferenceConfig,
    ) -> LLMMessage:
        """Prompt the LLM with a message and optional conversation history."""
        if len(messages) == 0:
            msg = "Must specify at least one message."
            raise ValueError(msg)
        request = self._create_request(messages, inference_cfg)
        response = self.invoke_model_with_request(request)
        return response.output.message.to_llm_message(inference_cfg)

    @timed_function
    def invoke_model_with_request(self, request: NovaInvokeLLMRequest) -> NovaResponse:
        """Invoke model with request and get a response back."""
        json_request = request.model_dump_json(exclude_none=True, by_alias=True)
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json_request,
            )
        except botocore.exceptions.ClientError:
            log.exception("Request body: %s", request.model_dump_json())
            raise
        response_body = response["body"].read().decode("UTF-8")
        nova_response = NovaResponse.model_validate_json(response_body)
        log.info("Token usage: %s", nova_response.usage)
        return nova_response


#########################
# Code for manual testing
# ruff: noqa: ERA001, T201

# llm = BedrockNovaLLM()
# config = LLMInferenceConfig()
# resp = llm.prompt(messages=[LLMMessage(content="hello")], inference_cfg=config)
# print(resp.model_dump_json(indent=2))


# llm = BedrockNovaLLM()
# config = LLMInferenceConfig(json_mode=True)
# resp = llm.prompt([LLMMessage(content="Create a list of the numbers 1 through 5")], config)
# print(resp.model_dump_json(indent=2))
