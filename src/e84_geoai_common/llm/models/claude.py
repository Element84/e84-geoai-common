import logging
from collections.abc import Sequence
from typing import Any, Literal

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

# See https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
ANTHROPIC_API_VERSION = "bedrock-2023-05-31"

# https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html#model-ids-arns
CLAUDE_BEDROCK_MODEL_IDS = {
    "Claude 3 Haiku": "us.anthropic.claude-3-haiku-20240307-v1:0",
    "Claude 3.5 Sonnet": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    "Claude 3 Sonnet": "us.anthropic.claude-3-sonnet-20240229-v1:0",
    "Claude 3 Opus": "us.anthropic.claude-3-opus-20240229-v1:0",
    "Claude Instant": "us.anthropic.claude-instant-v1",
    "Claude 3.5 Haiku": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "Claude 3.5 Sonnet v2": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
}

ConverseMediaTypeFormat = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]


class ClaudeTextContent(BaseModel):
    """Claude text context model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["text"] = "text"
    text: str


class ClaudeImageSource(BaseModel):
    """An image encoded for communication with an LLM."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["base64"] = "base64"
    media_type: ConverseMediaTypeFormat
    data: str


class ClaudeImageContent(BaseModel):
    """Claude text context model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["image"] = "image"
    source: ClaudeImageSource


class ClaudeMessage(BaseModel):
    """Claude message base model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    role: Literal["assistant", "user"]
    content: str | Sequence[ClaudeTextContent | ClaudeImageContent]

    @staticmethod
    def from_llm_message(msg: LLMMessage) -> "ClaudeMessage":
        """Converts the generic LLM Message into a ClaudeMessage."""

        def _handle_content(
            subcontent: TextContent | Base64ImageContent,
        ) -> ClaudeTextContent | ClaudeImageContent:
            if isinstance(subcontent, TextContent):
                return ClaudeTextContent(text=subcontent.text)
            return ClaudeImageContent(
                source=ClaudeImageSource(media_type=subcontent.media_type, data=subcontent.data)
            )

        if isinstance(msg.content, str):
            content = msg.content
        else:
            content = [_handle_content(subcontent) for subcontent in msg.content]
        return ClaudeMessage(role=msg.role, content=content)


class ClaudeToolChoice(BaseModel):
    """Claude tool choice model."""

    type: Literal["auto", "any", "tool"]
    name: str | None = None
    # disable_parallel_tool_use is documented in Anthropic docs but seems to
    # not be supported in Bedrock
    # disable_parallel_tool_use: bool | None = None # noqa: ERA001


class ClaudeTool(BaseModel):
    """Representation of a tool that Claude can use."""

    name: str
    description: str | None = None
    input_schema: dict[str, Any]


class ClaudeInvokeLLMRequest(BaseModel):
    """Represents a request to invoke Claude and get a response back."""

    model_config = ConfigDict(strict=True, extra="forbid")

    anthropic_version: str = ANTHROPIC_API_VERSION

    max_tokens: int = Field(default=1000, description="Maximum number of output tokens")

    messages: list[ClaudeMessage] = Field(default_factory=list, description="List of LLM Messages")

    stop_sequences: list[str] | None = None

    system: str | None = Field(default=None, description="System Prompt")

    temperature: float = Field(
        default=0,
        description="Temperature control for randomness. Closer to zero = more deterministic.",
    )

    tool_choice: ClaudeToolChoice | None = Field(
        default=None,
        description="Whether the model should use a specific "
        "tool, or any tool, or decide by itself.",
    )

    tools: list[ClaudeTool] | None = Field(
        default=None, description="List of tools that the model may call."
    )

    top_k: int | None = Field(default=None, description="Top K for sampling")

    top_p: float | None = Field(default=None, description="Top P for nucleus sampling")


#################################################################################
# Response objects


class ClaudeTextResponse(BaseModel):
    """Represents a text use response from Claude."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    type: Literal["text"] = "text"
    text: str


class ClaudeToolUseResponse(BaseModel):
    """Represents a tool use response from Claude."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    type: Literal["tool_use"] = "tool_use"
    id: str
    input: dict[str, Any]
    name: str


class ClaudeUsageInfo(BaseModel):
    """Claude usage-info model."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    input_tokens: int
    output_tokens: int


class ClaudeResponse(BaseModel):
    """Claude response model."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    content: Sequence[ClaudeTextResponse | ClaudeToolUseResponse]
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    stop_sequence: str | None = None
    type: Literal["message"]
    usage: ClaudeUsageInfo


def _config_to_response_prefix(config: LLMInferenceConfig) -> str | None:
    if config.json_mode:
        return "{"
    if config.response_prefix:
        return config.response_prefix
    return None


class BedrockClaudeLLM(LLM):
    """Implements the LLM class for Bedrock Claude."""

    client: BedrockRuntimeClient

    def __init__(
        self,
        model_id: str = CLAUDE_BEDROCK_MODEL_IDS["Claude 3 Haiku"],
        client: BedrockRuntimeClient | None = None,
    ) -> None:
        """Initialize.

        Args:
            model_id: Model ID. Defaults to the model ID for Claude 3 Haiku.
            client: Optional pre-initialized boto3 client. Defaults to None.
        """
        self.model_id = model_id
        self.client = client or boto3.client("bedrock-runtime")  # type: ignore[reportUnknownMemberType]

    def _create_request(
        self, messages: Sequence[LLMMessage], config: LLMInferenceConfig
    ) -> ClaudeInvokeLLMRequest:
        response_prefix = _config_to_response_prefix(config)
        if response_prefix:
            messages = [*messages, LLMMessage(role="assistant", content=response_prefix)]

        return ClaudeInvokeLLMRequest(
            max_tokens=config.max_tokens,
            system=config.system_prompt,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            messages=[ClaudeMessage.from_llm_message(msg) for msg in messages],
        )

    @timed_function
    def prompt(
        self,
        messages: Sequence[LLMMessage],
        inference_cfg: LLMInferenceConfig,
    ) -> LLMMessage:
        """Prompt the LLM with a message and optional conversation history."""
        if len(messages) == 0:
            raise ValueError("Must specify at least one message.")
        request = self._create_request(messages, inference_cfg)
        response = self.invoke_model_with_request(request)

        def _response_content_to_text(
            index: int, c: ClaudeTextResponse | ClaudeToolUseResponse
        ) -> TextContent:
            if isinstance(c, ClaudeToolUseResponse):
                raise TypeError("Did not expect a tool use response")
            text = c.text

            response_prefix = _config_to_response_prefix(inference_cfg)
            if index == 0 and response_prefix:
                text = response_prefix + text

            return TextContent(text=text)

        return LLMMessage(
            role="assistant",
            content=[_response_content_to_text(i, c) for i, c in enumerate(response.content)],
        )

    @timed_function
    def invoke_model_with_request(self, request: ClaudeInvokeLLMRequest) -> ClaudeResponse:
        """Invoke model with request and get a response back."""
        try:
            response = self.client.invoke_model(
                modelId=self.model_id, body=request.model_dump_json(exclude_none=True)
            )
        except botocore.exceptions.ClientError:
            log.exception("Request body: %s", request.model_dump_json())
            raise
        response_body = response["body"].read().decode("UTF-8")
        claude_response = ClaudeResponse.model_validate_json(response_body)
        log.info("Token usage: %s", claude_response.usage)
        return claude_response

    # FUTURE implement tool use
