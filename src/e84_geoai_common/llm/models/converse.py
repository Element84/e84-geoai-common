import base64
import json
import logging
from collections.abc import Sequence
from typing import Any, Literal, Self, cast

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

# Converse uses camel case for its variables. Ignore any linting problems with this.
# ruff: noqa: N815

log = logging.getLogger(__name__)


CONVERSE_BEDROCK_MODEL_IDS = {
    "Claude 3 Haiku": "us.anthropic.claude-3-haiku-20240307-v1:0",
    "Claude 3.5 Sonnet": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    "Claude 3 Sonnet": "us.anthropic.claude-3-sonnet-20240229-v1:0",
    "Claude 3 Opus": "anthropic.claude-3-opus-20240229-v1:0",
    "Claude Instant": "anthropic.claude-instant-v1",
    "Claude 3.5 Haiku": "anthropic.claude-3-5-haiku-20241022-v1:0",
    "Claude 3.5 Sonnet v2": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "Nova Lite": "us.amazon.nova-lite-v1:0",
    "Nova Micro": "us.amazon.nova-micro-v1:0",
    "Nova Pro": "us.amazon.nova-pro-v1:0",
    "Llama 3.1 70B Instruct": "us.meta.llama3-1-70b-instruct-v1:0",
    "Llama 3.1 8B Instruct": "us.meta.llama3-1-8b-instruct-v1:0",
    "Llama 3.2 11B Vision Instruct": "us.meta.llama3-2-11b-instruct-v1:0",
    "Llama 3.2 1B Instruct": "us.meta.llama3-2-1b-instruct-v1:0",
    "Llama 3.2 3B Instruct": "us.meta.llama3-2-3b-instruct-v1:0",
    "Llama 3.2 90B Vision Instruct": "us.meta.llama3-2-90b-instruct-v1:0",
    "Llama 3.3 70B Instruct": "us.meta.llama3-3-70b-instruct-v1:0"
}


#################################################################################
# Messages Object Components


class ConverseTextContent(BaseModel):
    """Converse text context model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    text: str

    def __str__(self) -> str:
        return self.text


class ConverseToolUseInnerContent(BaseModel):
    """Converse tool-use request model, inner."""

    model_config = ConfigDict(strict=True, extra="forbid")

    toolUseId: str
    name: str
    input: dict[str, Any]


class ConverseToolUseContent(BaseModel):
    """Converse tool-use request model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    toolUse: ConverseToolUseInnerContent


class ConverseToolResultInnerContent(BaseModel):
    """Converse tool inner result model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    toolUseId: str
    content: list[dict[str, Any]]
    status: str | None = None


class ConverseToolResultContent(BaseModel):
    """Converse tool result modoel."""

    model_config = ConfigDict(strict=True, extra="forbid")

    toolResult: ConverseToolResultInnerContent


class ConverseImageSource(BaseModel):

    model_config = ConfigDict(strict=True, extra="forbid")
    bytes: bytes


class ConverseImage(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    format: Literal["jpeg", "png", "gif", "webp"]
    source: ConverseImageSource


class ConverseImageContent(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    image: ConverseImage

    @classmethod
    def from_b64_image_content(cls, image: Base64ImageContent) -> Self:
        img_format: Literal["jpeg", "png", "gif", "webp"] = cast(
        Literal["jpeg", "png", "gif", "webp"], image.media_type.split("/")[-1]
    )
        source=ConverseImageSource(bytes=base64.b64decode(image.data))
        return cls(
            image = ConverseImage(
                format=img_format,
                source=source
            )
        )

    def to_b64_image_content(self) -> Base64ImageContent:
        media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"] = cast(
            Literal["image/jpeg", "image/png", "image/gif", "image/webp"],
            f"image/{self.image.format}"
        )
        return Base64ImageContent(
            media_type=media_type,
            data=self.image.source.bytes.decode("utf8"),
        )


class ConverseMessage(BaseModel):
    """Converse base model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    role: Literal["assistant", "user"]
    content: Sequence[
            ConverseTextContent | ConverseImageContent
        ]


    @classmethod
    def from_llm_message(cls, msg: LLMMessage) -> Self:
        def _handle_content(
            subcontent: TextContent | Base64ImageContent,
        ) -> ConverseTextContent | ConverseImageContent:
            if isinstance(subcontent, TextContent):
                return ConverseTextContent(text=subcontent.text)
            return ConverseImageContent.from_b64_image_content(subcontent)

        if isinstance(msg.content, str):
            content = [ConverseTextContent(text=msg.content)]
        else:
            content = [_handle_content(subcontent) for subcontent in msg.content]
        return cls(role=msg.role, content=content)

    def to_llm_message(self, inference_cfg: LLMInferenceConfig) -> LLMMessage:
        def _to_llm_content(
            index: int,
            c: ConverseTextContent | ConverseImageContent,
        ) -> TextContent | Base64ImageContent:
            if isinstance(c, ConverseTextContent):
                content = c.text
                if index == 0 and inference_cfg.response_prefix:
                    content = inference_cfg.response_prefix + content
                return TextContent(text=content)
            return c.to_b64_image_content()

        if len(self.content) == 1 and isinstance(self.content[0], ConverseTextContent):
            content = self.content[0].text
            if inference_cfg.json_mode:
                # In JSON mode we need to remove the JSON stop sequence
                content = [TextContent(text = "{" + content)]
            elif inference_cfg.response_prefix:
                content = [TextContent(text = inference_cfg.response_prefix + content)]
            else:
                content = [TextContent(text = content)]
        else:
            content = [_to_llm_content(index, c) for index, c in enumerate(self.content)]
        return LLMMessage(role=self.role, content=content)


class ConverseUserMessage(ConverseMessage):
    """Converse user message model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    role: Literal["user"] = "user"
    content: list[ConverseTextContent | ConverseToolResultContent]


class ConverseAssistantMessage(ConverseMessage):
    """Converse assistant message model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    role: Literal["assistant"] = "assistant"
    content: list[ConverseTextContent | ConverseToolUseContent]


#################################################################################
# Other Request Objects


class ConverseSingleTool(BaseModel):
    """Converse single tool model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    toolSpec: dict[str, Any]


class ConverseTools(BaseModel):
    """Converse tools model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    tools: Sequence[ConverseSingleTool]

class ConverseToolChoice(BaseModel):
    """Converse tool choice model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    #only anthropic and mistral models use tool choice. It is not implemented
    #in the code here yet - this is a dummy class.
    type: Literal["auto", "any", "tool"]
    name: str | None = None
    # disable_parallel_tool_use is documented in Anthropic docs but seems to
    # not be supported in Bedrock
    # disable_parallel_tool_use: bool | None = None  # noqa: ERA001


class SystemContentBlock(BaseModel):
    """A system prompt block."""

    model_config = ConfigDict(strict=True, extra="forbid")

    text: str


class ConverseInferenceConfig(BaseModel):
    """Converse inference config model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    maxTokens: int | None
    stopSequences: Sequence[str] | None
    temperature: float | None
    topP: float | None


class ConverseAdditionalModelRequestFields(BaseModel):
    """Converse additional fields for certain models."""

    model_config = ConfigDict(strict=True, extra="forbid")

    top_k: int | None


class ConverseInvokeLLMRequest(BaseModel):
    """Converse request model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    additionalModelRequestFields: ConverseAdditionalModelRequestFields | None = Field(
        default=None, description="Configuration for any additionoal fields models may need"
    )

    inferenceConfig: ConverseInferenceConfig | None = Field(
        default=None, description="Configuration of maxTokens, stopSequences, temperature, topP"
    )

    modelId: str = Field(
        default=CONVERSE_BEDROCK_MODEL_IDS["Claude 3 Haiku"],
        description="Model used for the Converse api"
    )

    messages: list[ConverseMessage] = Field(
        default_factory=list, description="List of LLM Messages"
    )

    system: Sequence[SystemContentBlock] | None = Field(default=None, description="System Prompt")

    toolConfig: ConverseTools | None  = Field(
        default=None, description="List of tools that the model may call."
    )

#################################################################################
# Response objects

class ConverseUsageInfo(BaseModel):
    """Usage info from the Converse API."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    inputTokens: int
    outputTokens: int
    totalTokens: int

class ConverseMessageResponse(BaseModel):

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    message: ConverseAssistantMessage

class ConverseMetrics(BaseModel):

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    latencyMs: int

class ConverseResponse(BaseModel):
    """Converse response model."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    additionalModelResponseFields: dict[str, Any] | None = Field(default = None)
    metrics: ConverseMetrics
    output: ConverseMessageResponse
    performanceConfig: dict[str, Any] | None = Field(default = None)
    ResponseMetadata: dict[str, Any]
    role: Literal["assistant"] = "assistant"
    stopReason: Literal[
        "end_turn", "max_tokens", "stop_sequence", "tool_use"
    ]
    trace: dict[str, Any] | None = Field(default = None)
    usage: ConverseUsageInfo

def _config_to_response_prefix(config: LLMInferenceConfig) -> str | None:
    if config.json_mode:
        return "{"
    if config.response_prefix:
        return config.response_prefix
    return None

class BedrockConverseLLM(LLM):
    """Implements the LLM class for Bedrock Converse."""
    client: BedrockRuntimeClient

    def __init__(
        self,
        model_id: str = CONVERSE_BEDROCK_MODEL_IDS["Claude 3 Haiku"],
        client: BedrockRuntimeClient | None = None,
    ) -> None:
        """Initialize.

        Args:
            model_id: Model ID. Defaults to the model ID for Claude 3 Haiku.
            client: Optional pre-initialized boto3 client. Defaults to None.
        """
        self.model_id = model_id
        self.client = client or boto3.client("bedrock-runtime") # type: ignore[reportUnknownMemberType]

    def _create_request(
        self, messages: Sequence[LLMMessage], config: LLMInferenceConfig
    ) -> ConverseInvokeLLMRequest:

            response_prefix = _config_to_response_prefix(config)
            if response_prefix:
                messages = [*messages, LLMMessage(role="assistant", content=response_prefix)]
            system = None
            if config.system_prompt:
                #For now, just support for a single system prompt content block
                system = [SystemContentBlock(text=config.system_prompt)]
            stop_sequences = None #stop sequence not implemented yet
            tools = None

            inference_config = ConverseInferenceConfig(
                maxTokens = config.max_tokens,
                stopSequences=stop_sequences,
                temperature=config.temperature,
                topP=config.top_p
            )

            additional_model_request_fields = ConverseAdditionalModelRequestFields(
                top_k=config.top_k
            )
            if not additional_model_request_fields.model_dump(exclude_none=True):
                additional_model_request_fields = None

            request = ConverseInvokeLLMRequest(
                modelId=self.model_id,
                messages=[ConverseMessage.from_llm_message(msg) for msg in messages],
                toolConfig=tools,
                system=system,
                inferenceConfig=inference_config,
                additionalModelRequestFields=additional_model_request_fields
            )

            return request


    @timed_function
    def prompt(
        self,
        messages: Sequence[LLMMessage],
        inference_cfg: LLMInferenceConfig,
    ) -> LLMMessage:
        """Prompt the LLM with a message and optional conversation history."""
        if not messages:
            msg = "Must specify at least one message."
            raise ValueError(msg)
        request = self._create_request(messages = messages, config=inference_cfg)
        response = self.invoke_model_with_request(request)
        return response.output.message.to_llm_message(inference_cfg)


    @timed_function
    def invoke_model_with_request(
        self, request: ConverseInvokeLLMRequest
    ) -> ConverseResponse:
        """Invoke model with request and get a response back."""
        response_body = self._make_client_request(request)
        raw_data = json.loads(response_body)
        response = ConverseResponse.model_validate(raw_data)
        return response

    def _make_client_request(self, request: ConverseInvokeLLMRequest) -> str:
        """Make model invocation request and return raw JSON response."""
        try:
            params = request.model_dump(exclude_none=True)
            response = self.client.converse(**params)
        except botocore.exceptions.ClientError as e:
            log.error("Failed with %s", e)  # noqa: TRY400
            log.error("Request body: %s", request)  # noqa: TRY400
            raise
        return json.dumps(response)

    def _add_prefix_to_response(
        self, response: ConverseResponse, prefix: str
    ) -> ConverseResponse:
        """Prepend the prefix to the first text block in the response."""
        for content_block in response.output.message.content:
            if isinstance(content_block, ConverseTextContent):
                content_block.text = prefix + content_block.text
                break
        return response
