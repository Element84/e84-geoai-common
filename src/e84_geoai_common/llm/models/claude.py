import json
import logging
from collections.abc import Sequence
from itertools import pairwise
from typing import Any, Literal, cast

import boto3
import botocore.config
import botocore.exceptions
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from pydantic import BaseModel, ConfigDict, Field

from e84_geoai_common.llm.core.llm import (
    LLM,
    Base64ImageContent,
    CachePointContent,
    JSONContent,
    LLMAssistantMessage,
    LLMDataContentType,
    LLMInferenceConfig,
    LLMMessage,
    LLMResponseMetadata,
    LLMTool,
    LLMToolChoice,
    LLMToolResultContent,
    LLMToolUseContent,
    TextContent,
)
from e84_geoai_common.tracing import observe, timed_function, update_current_generation

log = logging.getLogger(__name__)

# See https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
ANTHROPIC_API_VERSION = "bedrock-2023-05-31"

CLAUDE_3_HAIKU = "us.anthropic.claude-3-haiku-20240307-v1:0"
CLAUDE_3_5_SONNET = "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
CLAUDE_3_SONNET = "us.anthropic.claude-3-sonnet-20240229-v1:0"
CLAUDE_3_OPUS = "us.anthropic.claude-3-opus-20240229-v1:0"
CLAUDE_INSTANT = "us.anthropic.claude-instant-v1"
CLAUDE_3_5_HAIKU = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
CLAUDE_3_5_SONNET_V2 = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
CLAUDE_3_7_SONNET = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
CLAUDE_4_SONNET = "us.anthropic.claude-sonnet-4-20250514-v1:0"
CLAUDE_4_OPUS = "us.anthropic.claude-opus-4-20250514-v1:0"
CLAUDE_4_1_OPUS = "us.anthropic.claude-opus-4-1-20250805-v1:0"
CLAUDE_4_5_HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
CLAUDE_4_5_SONNET = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


_MODELS_NOT_SUPPORTING_CACHING = {
    CLAUDE_3_HAIKU,
    CLAUDE_3_5_SONNET,
    CLAUDE_3_SONNET,
    CLAUDE_3_OPUS,
    CLAUDE_INSTANT,
    CLAUDE_3_5_SONNET_V2,
}


def _supports_caching(model: str) -> bool:
    """Returns true if the model supports caching."""
    return model not in _MODELS_NOT_SUPPORTING_CACHING


# https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html#model-ids-arns
# DEPRECATED: Use the constants above instead.
CLAUDE_BEDROCK_MODEL_IDS = {
    "Claude 3 Haiku": CLAUDE_3_HAIKU,
    "Claude 3.5 Sonnet": CLAUDE_3_5_SONNET,
    "Claude 3 Sonnet": CLAUDE_3_SONNET,
    "Claude 3 Opus": CLAUDE_3_OPUS,
    "Claude Instant": CLAUDE_INSTANT,
    "Claude 3.5 Haiku": CLAUDE_3_5_HAIKU,
    "Claude 3.5 Sonnet v2": CLAUDE_3_5_SONNET_V2,
    "Claude 3.7 Sonnet": CLAUDE_3_7_SONNET,
    "Claude 4 Sonnet": CLAUDE_4_SONNET,
    # We won't add any new models to this as we want to remove it eventually.
}

ConverseMediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]


class ClaudeCacheControl(BaseModel, frozen=True):
    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["ephemeral"] = "ephemeral"


class ClaudeCacheableContent(BaseModel):
    cache_control: ClaudeCacheControl | None = Field(default=None)


class ClaudeTextContent(ClaudeCacheableContent):
    """Claude text context model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["text"] = "text"
    text: str


class ClaudeImageSource(ClaudeCacheableContent):
    """An image encoded for communication with an LLM."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["base64"] = "base64"
    media_type: ConverseMediaType
    data: str


class ClaudeImageContent(ClaudeCacheableContent):
    """Claude text context model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["image"] = "image"
    source: ClaudeImageSource


class ClaudeToolUseContent(ClaudeCacheableContent):
    """Represents a tool use request from Claude."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    type: Literal["tool_use"] = "tool_use"
    id: str
    input: dict[str, Any]
    name: str


class ClaudeToolResultContent(ClaudeCacheableContent):
    """Claude tool result model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str | Sequence[ClaudeTextContent | ClaudeImageContent]
    is_error: bool | None = None


ClaudeMessageContentType = (
    ClaudeTextContent | ClaudeImageContent | ClaudeToolUseContent | ClaudeToolResultContent
)


class ClaudeMessage(BaseModel):
    """Claude message base model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    role: Literal["assistant", "user"]
    content: str | Sequence[ClaudeMessageContentType]


class ClaudeToolChoice(BaseModel):
    """Claude tool choice model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["auto", "any", "tool"]
    name: str | None = None
    # disable_parallel_tool_use is documented in Anthropic docs but seems to
    # not be supported in Bedrock
    # disable_parallel_tool_use: bool | None = None # noqa: ERA001


class ClaudeTool(ClaudeCacheableContent):
    """Representation of a tool that Claude can use."""

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str
    description: str | None = None
    input_schema: dict[str, Any]


class ClaudeInvokeLLMRequest(BaseModel, frozen=True):
    """Represents a request to invoke Claude and get a response back."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    anthropic_version: str = ANTHROPIC_API_VERSION

    max_tokens: int = Field(default=1000, description="Maximum number of output tokens")

    messages: list[ClaudeMessage] = Field(
        default_factory=list[ClaudeMessage], description="List of LLM Messages"
    )

    stop_sequences: list[str] | None = None

    system: Sequence[ClaudeMessageContentType] | None = Field(
        default=None, description="System Prompt"
    )

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
# Other response objects


class ClaudeUsageInfo(BaseModel, frozen=True):
    """Claude usage-info model."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class ClaudeResponse(BaseModel, frozen=True):
    """Claude response model."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    content: Sequence[ClaudeTextContent | ClaudeToolUseContent]
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    stop_sequence: str | None = None
    type: Literal["message"]
    usage: ClaudeUsageInfo


def _llm_message_to_claude_message(msg: LLMMessage) -> "ClaudeMessage":
    """Converts the generic LLM Message into a ClaudeMessage."""

    def _handle_content(
        content: TextContent | Base64ImageContent | LLMToolUseContent | LLMToolResultContent,
        *,
        cache: bool = False,
    ) -> ClaudeMessageContentType:
        cache_control = ClaudeCacheControl() if cache else None
        match content:
            case TextContent():
                return ClaudeTextContent(text=content.text, cache_control=cache_control)
            case Base64ImageContent():
                return ClaudeImageContent(
                    source=ClaudeImageSource(
                        media_type=content.media_type,
                        data=content.data,
                        cache_control=cache_control,
                    )
                )
            case LLMToolUseContent():
                return ClaudeToolUseContent(
                    id=content.id,
                    name=content.name,
                    input=content.input,
                    cache_control=cache_control,
                )
            case LLMToolResultContent():
                return _llm_tool_result_to_claude_tool_result(content, cache_control=cache_control)

    if isinstance(msg.content, str):
        content = [ClaudeTextContent(type="text", text=msg.content, cache_control=None)]
    else:
        content = [
            _handle_content(content, cache=isinstance(next_content, CachePointContent))
            for content, next_content in pairwise([*msg.content, None])
            if content and not isinstance(content, CachePointContent)
        ]
    return ClaudeMessage(role=msg.role, content=content)


def _llm_tool_to_claude_tool(tool: LLMTool) -> ClaudeTool:
    """Build a ClaudeTool from an LLMTool.

    If LLMTool.output_model is set, the JSON schema of the output model is
    injected into the tool's description so that the LLM is aware of it.
    """
    if tool.input_model is None:
        input_schema = cast("dict[str, Any]", {"type": "object", "properties": {}})
    else:
        input_schema = tool.input_model.model_json_schema()

    description = tool.description
    if tool.output_model is not None:
        output_schema = json.dumps(tool.output_model.model_json_schema())
        description = f"{tool.description}\n\nOutput schema:\n```json\n{output_schema}\n```"

    claude_tool = ClaudeTool(
        name=tool.name,
        description=description,
        input_schema=input_schema,
    )
    return claude_tool


def _llm_tool_choice_to_claude_tool_choice(
    tool_choice: LLMToolChoice | None,
) -> ClaudeToolChoice:
    if tool_choice is None:
        return ClaudeToolChoice(type="auto")
    match tool_choice.mode:
        case "optional":
            return ClaudeToolChoice(type="auto")
        case "force_tool_use":
            return ClaudeToolChoice(type="any")
        case "force_specific_tool_use":
            return ClaudeToolChoice(type="tool", name=tool_choice.tool_name)


def _llm_tool_result_to_claude_tool_result(
    tool_result: LLMToolResultContent, cache_control: ClaudeCacheControl | None = None
) -> ClaudeToolResultContent:
    def _to_tool_result_content(
        in_content: LLMDataContentType,
    ) -> ClaudeTextContent | ClaudeImageContent:
        match in_content:
            case TextContent():
                out_content = ClaudeTextContent(text=in_content.text)
            case JSONContent():
                text = json.dumps(in_content.data)
                out_content = ClaudeTextContent(text=text)
            case Base64ImageContent():
                out_content = ClaudeImageContent(
                    source=ClaudeImageSource(media_type=in_content.media_type, data=in_content.data)
                )
        return out_content

    out_content = [_to_tool_result_content(c) for c in tool_result.content]
    out = ClaudeToolResultContent(
        tool_use_id=tool_result.id,
        is_error=(tool_result.status == "error"),
        content=out_content,
        cache_control=cache_control,
    )
    return out


class BedrockClaudeLLM(LLM):
    """Implements the LLM class for Bedrock Claude."""

    client: BedrockRuntimeClient

    def __init__(
        self,
        model_id: str = CLAUDE_3_5_HAIKU,
        client: BedrockRuntimeClient | None = None,
    ) -> None:
        """Initialize.

        Args:
            model_id: Model ID. Defaults to the model ID for Claude 3 Haiku.
            client: Optional pre-initialized boto3 client. Defaults to None.
        """
        self.model_id = model_id
        self.client = client or boto3.client(  # type: ignore[reportUnknownMemberType]
            "bedrock-runtime",
            config=botocore.config.Config(read_timeout=300),
        )

    def create_request(
        self, messages: Sequence[LLMMessage], config: LLMInferenceConfig
    ) -> ClaudeInvokeLLMRequest:
        stop_sequences = None
        if config.json_mode:
            # https://docs.aws.amazon.com/nova/latest/userguide/prompting-structured-output.html
            prefix = "```json\n{"
            messages = [*messages, LLMMessage(role="assistant", content=prefix)]
            stop_sequences = ["```"]
        elif config.response_prefix:
            messages = [*messages, LLMMessage(role="assistant", content=config.response_prefix)]

        tools = None
        tool_choice = None
        if config.tools is not None:
            tools = [_llm_tool_to_claude_tool(t) for t in config.tools]
            tool_choice = _llm_tool_choice_to_claude_tool_choice(config.tool_choice)

            if _supports_caching(self.model_id):
                # If the model supports caching we'll cache all of the tool definitions by default.
                # Everyt tool before the last tool will be included in this cache.
                last_tool = tools[-1]
                last_tool.cache_control = ClaudeCacheControl()

        system_content: list[ClaudeMessageContentType] | None = None

        if config.system_prompt:
            cache_control = ClaudeCacheControl() if _supports_caching(self.model_id) else None
            system_content = [
                ClaudeTextContent(cache_control=cache_control, text=config.system_prompt)
            ]

        return ClaudeInvokeLLMRequest(
            max_tokens=config.max_tokens,
            system=system_content,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            tools=tools,
            tool_choice=tool_choice,
            stop_sequences=stop_sequences,
            messages=[_llm_message_to_claude_message(msg) for msg in messages],
        )

    @timed_function
    @observe(as_type="generation", name="BedrockClaudeLLM.prompt")
    def prompt(
        self,
        messages: Sequence[LLMMessage],
        inference_cfg: LLMInferenceConfig,
    ) -> LLMAssistantMessage:
        """Prompt the LLM with a message and optional conversation history."""
        if len(messages) == 0:
            raise ValueError("Must specify at least one message.")
        request = self.create_request(messages, inference_cfg)
        response = self.invoke_model_with_request(request)
        llm_msg = self._response_to_llm_message(response, inference_cfg=inference_cfg)
        return llm_msg

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
        update_current_generation(
            usage_details={
                "input": claude_response.usage.input_tokens,
                "output": claude_response.usage.output_tokens,
                "cache_read_input_tokens": claude_response.usage.cache_read_input_tokens,
                "cache_creation_input_tokens": claude_response.usage.cache_creation_input_tokens,
            }
        )
        return claude_response

    def _response_to_llm_message(
        self, response: ClaudeResponse, inference_cfg: LLMInferenceConfig
    ) -> LLMAssistantMessage:
        def _to_llm_content(
            index: int, c: ClaudeTextContent | ClaudeImageContent | ClaudeToolUseContent
        ) -> TextContent | Base64ImageContent | LLMToolUseContent:
            match c:
                case ClaudeTextContent():
                    text = c.text
                    if index == 0:
                        if inference_cfg.json_mode:
                            text = "{" + text.removesuffix("```")
                        elif inference_cfg.response_prefix:
                            text = inference_cfg.response_prefix + text
                    return TextContent(text=text)
                case ClaudeImageContent():
                    return Base64ImageContent(media_type=c.source.media_type, data=c.source.data)
                case ClaudeToolUseContent():
                    return LLMToolUseContent(id=c.id, name=c.name, input=c.input)

        return LLMAssistantMessage(
            content=[_to_llm_content(i, c) for i, c in enumerate(response.content)],
            metadata=LLMResponseMetadata(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                stop_reason=response.stop_reason,
            ),
        )
