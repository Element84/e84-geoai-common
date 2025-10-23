import logging
from collections.abc import Sequence
from functools import reduce
from typing import Literal, Self, cast

import boto3
import botocore.exceptions
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from pydantic import BaseModel, ConfigDict, Field

from e84_geoai_common.llm.core.llm import (
    LLM,
    Base64ImageContent,
    CachePointContent,
    LLMAssistantMessage,
    LLMInferenceConfig,
    LLMMediaType,
    LLMMessage,
    LLMMessageContentType,
    LLMResponseMetadata,
    LLMToolResultContent,
    LLMToolUseContent,
    TextContent,
)
from e84_geoai_common.tracing import observe, timed_function

log = logging.getLogger(__name__)


NOVA_CANVAS = "us.amazon.nova-canvas-v1:0"
NOVA_LITE = "us.amazon.nova-lite-v1:0"
NOVA_MICRO = "us.amazon.nova-micro-v1:0"
NOVA_PRO = "us.amazon.nova-pro-v1:0"
NOVA_REEL = "us.amazon.nova-reel-v1:0"

# https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html#model-ids-arns
# DEPRECATED: Use the constants above instead.
NOVA_BEDROCK_MODEL_IDS = {
    "Nova Canvas": NOVA_CANVAS,
    "Nova Lite": NOVA_LITE,
    "Nova Micro": NOVA_MICRO,
    "Nova Pro": NOVA_PRO,
    "Nova Reel": NOVA_REEL,
}

NovaImageFormat = Literal["jpeg", "png", "gif", "webp"]


class NovaCachePoint(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["default"] = "default"


class NovaCacheableContent(BaseModel):
    cache_point: NovaCachePoint | None = Field(default=None)


class NovaTextContent(NovaCacheableContent):
    """Nova text context model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    text: str


class NovaImageSource(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    bytes: str


class NovaImageInnerContent(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    format: NovaImageFormat
    source: NovaImageSource


class NovaImageContent(NovaCacheableContent):
    model_config = ConfigDict(strict=True, extra="forbid")

    image: NovaImageInnerContent

    @classmethod
    def from_b64_image_content(cls, image: Base64ImageContent) -> Self:
        img_format: NovaImageFormat = cast("NovaImageFormat", image.media_type.split("/")[-1])
        return cls(
            image=NovaImageInnerContent(
                format=img_format,
                source=NovaImageSource(bytes=image.data),
            )
        )

    def to_b64_image_content(self) -> Base64ImageContent:
        media_type: LLMMediaType = cast(
            "LLMMediaType",
            f"image/{self.image.format}",
        )
        return Base64ImageContent(
            media_type=media_type,
            data=self.image.source.bytes,
        )


class NovaMessage(BaseModel):
    """Nova message base model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    role: Literal["assistant", "user"] = "user"
    content: Sequence[NovaTextContent | NovaImageContent]


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

    messages: list[NovaMessage] = Field(default_factory=list[NovaMessage])

    inference_config: NovaInferenceConfig | None = Field(
        default=None, serialization_alias="inferenceConfig", alias="inferenceConfig"
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


def _llm_message_to_nova_message(msg: LLMMessage) -> NovaMessage:
    """Converts the generic LLM Message into a NovaMessage."""

    def _handle_content(
        acc: tuple[NovaTextContent | NovaImageContent, ...], content: LLMMessageContentType
    ) -> tuple[NovaTextContent | NovaImageContent, ...]:
        match content:
            case TextContent():
                return (*acc, NovaTextContent(text=content.text))
            case Base64ImageContent():
                return (*acc, NovaImageContent.from_b64_image_content(content))
            case LLMToolUseContent():
                raise NotImplementedError
            case LLMToolResultContent():
                raise NotImplementedError
            case CachePointContent():
                # If cache point is first element, drop it.
                if len(acc) == 0:
                    return acc

                # Modify should cache of last element in Sequence
                acc_until_last: tuple[NovaTextContent | NovaImageContent, ...] = acc[:-1]

                last_content = acc[-1]
                last_content.cache_point = NovaCachePoint()

                return (*acc_until_last, last_content)

    if isinstance(msg.content, str):
        content = [NovaTextContent(text=msg.content)]
    else:
        content = reduce(_handle_content, msg.content, ())
    return NovaMessage(role=msg.role, content=content)


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

    def create_request(
        self, messages: Sequence[LLMMessage], config: LLMInferenceConfig
    ) -> NovaInvokeLLMRequest:
        system = [NovaTextContent(text=config.system_prompt)] if config.system_prompt else None
        stop_sequences = None
        if config.json_mode:
            # https://docs.aws.amazon.com/nova/latest/userguide/prompting-structured-output.html
            prefix = "```json\n{"
            messages = [*messages, LLMMessage(role="assistant", content=prefix)]
            stop_sequences = ["```"]
        elif config.response_prefix:
            messages = [*messages, LLMMessage(role="assistant", content=config.response_prefix)]
        return NovaInvokeLLMRequest(
            system=system,
            inferenceConfig=NovaInferenceConfig(
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                stop_sequences=stop_sequences,
            ),
            messages=[_llm_message_to_nova_message(msg) for msg in messages],
        )

    @timed_function
    @observe(as_type="generation", name="BedrockNovaLLM.prompt")
    def prompt(
        self,
        messages: Sequence[LLMMessage],
        inference_cfg: LLMInferenceConfig,
    ) -> LLMAssistantMessage:
        """Prompt the LLM with a message and optional conversation history."""
        if len(messages) == 0:
            msg = "Must specify at least one message."
            raise ValueError(msg)
        request = self.create_request(messages, inference_cfg)
        response = self.invoke_model_with_request(request)
        llm_msg = self._response_to_llm_message(response, inference_cfg=inference_cfg)
        return llm_msg

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

    def _response_to_llm_message(
        self, response: NovaResponse, inference_cfg: LLMInferenceConfig
    ) -> LLMAssistantMessage:
        def _to_llm_content(
            index: int,
            c: NovaTextContent | NovaImageContent,
        ) -> TextContent | Base64ImageContent | LLMToolUseContent:
            match c:
                case NovaTextContent():
                    text = c.text
                    if index == 0:
                        if inference_cfg.json_mode:
                            text = "{" + text.removesuffix("```")
                        elif inference_cfg.response_prefix:
                            text = inference_cfg.response_prefix + text
                    return TextContent(text=text)
                case NovaImageContent():
                    return c.to_b64_image_content()

        response_msg = response.output.message
        content = [_to_llm_content(index, c) for index, c in enumerate(response_msg.content)]
        return LLMAssistantMessage(
            role="assistant",
            content=content,
            metadata=LLMResponseMetadata(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                stop_reason=response.stop_reason,
            ),
        )


#########################
# Code for manual testing
# ruff: noqa: ERA001

# llm = BedrockNovaLLM()
# config = LLMInferenceConfig()
# resp = llm.prompt(messages=[LLMMessage(content="hello")], inference_cfg=config)
# print(resp.model_dump_json(indent=2))


# llm = BedrockNovaLLM()
# config = LLMInferenceConfig(json_mode=True)
# resp = llm.prompt([LLMMessage(content="Create a list of the numbers 1 through 5")], config)
# print(resp.model_dump_json(indent=2))
