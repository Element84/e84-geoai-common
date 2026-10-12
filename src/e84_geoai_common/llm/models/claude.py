import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal, cast

import boto3
import botocore.config
import botocore.exceptions
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from pydantic import BaseModel, ConfigDict, Field, ValidationError

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
    LLMUserMessage,
    RedactedThinkingContent,
    TextContent,
    ThinkingContent,
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
CLAUDE_4_5_OPUS = "us.anthropic.claude-opus-4-5-20251101-v1:0"
CLAUDE_4_6_SONNET = "us.anthropic.claude-sonnet-4-6"
CLAUDE_4_6_OPUS = "us.anthropic.claude-opus-4-6-v1"
CLAUDE_4_7_OPUS = "us.anthropic.claude-opus-4-7"
CLAUDE_4_8_OPUS = "us.anthropic.claude-opus-4-8"
CLAUDE_5_OPUS = "us.anthropic.claude-opus-5"
CLAUDE_5_5_OPUS = "us.anthropic.claude-opus-5-5"
CLAUDE_5_SONNET = "us.anthropic.claude-sonnet-5"
CLAUDE_5_5_SONNET = "us.anthropic.claude-sonnet-5-5"
CLAUDE_5_5_HAIKU = "us.anthropic.claude-haiku-5-5"


@dataclass(frozen=True)
class ClaudeModelCapabilities:
    """Which request features a Claude model accepts."""

    caching: bool = True
    # temperature, top_p, and top_k
    sampling_params: bool = True
    # A tool_choice of "any" or "tool"
    forced_tool_choice: bool = True


_NO_SAMPLING_PARAMS = ClaudeModelCapabilities(sampling_params=False)

# Every model constant above must have an entry here. Keys are "us." model IDs.
_MODEL_CAPABILITIES: dict[str, ClaudeModelCapabilities] = {
    CLAUDE_3_HAIKU: ClaudeModelCapabilities(caching=False),
    CLAUDE_3_5_SONNET: ClaudeModelCapabilities(caching=False),
    CLAUDE_3_SONNET: ClaudeModelCapabilities(caching=False),
    CLAUDE_3_OPUS: ClaudeModelCapabilities(caching=False),
    CLAUDE_INSTANT: ClaudeModelCapabilities(caching=False),
    CLAUDE_3_5_HAIKU: ClaudeModelCapabilities(),
    CLAUDE_3_5_SONNET_V2: ClaudeModelCapabilities(caching=False),
    CLAUDE_3_7_SONNET: ClaudeModelCapabilities(),
    CLAUDE_4_SONNET: ClaudeModelCapabilities(),
    CLAUDE_4_OPUS: ClaudeModelCapabilities(),
    CLAUDE_4_1_OPUS: ClaudeModelCapabilities(),
    CLAUDE_4_5_HAIKU: ClaudeModelCapabilities(),
    CLAUDE_4_5_SONNET: ClaudeModelCapabilities(),
    CLAUDE_4_5_OPUS: ClaudeModelCapabilities(),
    CLAUDE_4_6_SONNET: ClaudeModelCapabilities(),
    CLAUDE_4_6_OPUS: ClaudeModelCapabilities(),
    CLAUDE_4_7_OPUS: _NO_SAMPLING_PARAMS,
    CLAUDE_4_8_OPUS: _NO_SAMPLING_PARAMS,
    CLAUDE_5_OPUS: _NO_SAMPLING_PARAMS,
    CLAUDE_5_5_OPUS: ClaudeModelCapabilities(sampling_params=False, forced_tool_choice=False),
    CLAUDE_5_SONNET: _NO_SAMPLING_PARAMS,
    CLAUDE_5_5_SONNET: ClaudeModelCapabilities(sampling_params=False, forced_tool_choice=False),
    CLAUDE_5_5_HAIKU: _NO_SAMPLING_PARAMS,
}


def _to_us_model_id(model_id: str) -> str:
    """Normalizes a Claude model ID to the "us." inference profile ID used as a key above.

    Handles cross-region inference profile IDs ("global.anthropic.x"), foundation model IDs
    without a region prefix ("anthropic.x"), and ARNs of either. Other IDs (e.g. application
    inference profile ARNs) are returned unchanged.
    """
    # ARNs end in "foundation-model/<id>" or "inference-profile/<id>"
    model_id = model_id.rsplit("/", 1)[-1]
    prefix, sep, rest = model_id.partition("anthropic.")
    if sep and (prefix == "" or (prefix.endswith(".") and "." not in prefix[:-1])):
        return f"us.anthropic.{rest}"
    return model_id


def get_claude_model_capabilities(model_id: str) -> ClaudeModelCapabilities | None:
    """Returns the capabilities of a known Claude model, or None if the model isn't known."""
    return _MODEL_CAPABILITIES.get(_to_us_model_id(model_id))


def _capabilities(model_id: str) -> ClaudeModelCapabilities:
    """Returns the model's capabilities, assuming everything is supported for unknown models."""
    capabilities = get_claude_model_capabilities(model_id)
    if capabilities is None:
        log.debug("Unknown Claude model %s. Assuming it supports all features.", model_id)
        return ClaudeModelCapabilities()
    return capabilities


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
    "Claude 4.1 Opus": CLAUDE_4_1_OPUS,
    "Claude 4.5 Opus": CLAUDE_4_5_OPUS,
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


class ClaudeThinkingContent(BaseModel, frozen=True):
    """A thinking block returned by models with thinking enabled.

    These must be sent back unchanged when continuing a turn, e.g. with a tool result.
    """

    model_config = ConfigDict(strict=True, extra="allow", frozen=True)

    type: Literal["thinking"]
    thinking: str
    signature: str | None = None


class ClaudeRedactedThinkingContent(BaseModel, frozen=True):
    """A thinking block whose contents are encrypted."""

    model_config = ConfigDict(strict=True, extra="allow", frozen=True)

    type: Literal["redacted_thinking"]
    data: str


ClaudeMessageContentType = (
    ClaudeTextContent
    | ClaudeImageContent
    | ClaudeToolUseContent
    | ClaudeToolResultContent
    | ClaudeThinkingContent
    | ClaudeRedactedThinkingContent
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


class JSONSchemaFormat(BaseModel):
    """JSON schema to be used for formatting the output."""

    type: Literal["json_schema"] = "json_schema"
    schema_: dict[str, Any] = Field(alias="schema")


class ClaudeOutputConfig(BaseModel):
    """Configuration for the output of a Claude response."""

    model_config = ConfigDict(strict=True, extra="forbid")

    format: JSONSchemaFormat


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

    temperature: float | None = Field(
        # Not sent by default because some models reject it. BedrockClaudeLLM.create_request sets
        # this for models that accept it.
        default=None,
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

    output_config: ClaudeOutputConfig | None = Field(
        default=None, description="Configuration for the output of the model."
    )


#################################################################################
# Other response objects


class ClaudeCacheCreationInfo(BaseModel, frozen=True):
    """Breakdown of cached tokens by TTL."""

    model_config = ConfigDict(strict=True, extra="allow", frozen=True)

    ephemeral_1h_input_tokens: int = Field(
        description="The number of input tokens used to create the 1 hour cache entry."
    )
    ephemeral_5m_input_tokens: int = Field(
        description="The number of input tokens used to create the 5 minute cache entry."
    )


class ClaudeUsageInfo(BaseModel, frozen=True):
    """Claude usage-info model."""

    model_config = ConfigDict(strict=True, extra="allow", frozen=True)

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_creation: ClaudeCacheCreationInfo | None = None


class ClaudeStopDetails(BaseModel, frozen=True):
    """Structured information about a refusal."""

    type: Literal["refusal"]
    category: str | None = Field(
        default=None,
        description="The policy category that triggered the refusal, e.g. 'cyber' or 'bio'.",
    )
    explanation: str | None = Field(default=None, description="Explanation for the refusal.")


class ClaudeResponse(BaseModel, frozen=True):
    """Claude response model."""

    model_config = ConfigDict(strict=True, extra="allow", frozen=True)

    content: Sequence[
        ClaudeTextContent
        | ClaudeToolUseContent
        | ClaudeThinkingContent
        | ClaudeRedactedThinkingContent
    ]
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    stop_reason: Literal[
        "end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"
    ]
    stop_details: ClaudeStopDetails | None = None
    stop_sequence: str | None = None
    type: Literal["message"]
    usage: ClaudeUsageInfo


def _llm_message_to_claude_message(msg: LLMMessage) -> "ClaudeMessage":
    """Converts the generic LLM Message into a ClaudeMessage."""

    def _handle_content(
        content: TextContent
        | Base64ImageContent
        | LLMToolUseContent
        | LLMToolResultContent
        | ThinkingContent
        | RedactedThinkingContent,
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
            case ThinkingContent():
                # Thinking blocks can't be cached directly.
                return ClaudeThinkingContent(
                    type="thinking", thinking=content.thinking, signature=content.signature
                )
            case RedactedThinkingContent():
                return ClaudeRedactedThinkingContent(type="redacted_thinking", data=content.data)

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
        if "type" not in input_schema or any(
            k in input_schema for k in ("oneOf", "anyOf", "allOf")
        ):
            raise ValueError(
                "Tool input models must be JSON objects (i.e. model_json_schema must have "
                '"type": "object") and must not use oneOf/anyOf/allOf at the root. '
                "This usually happens when input_model is a union type. Consider using a custom "
                "root model with a single field if you want to use a Union."
            )

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
    tool_choice: LLMToolChoice | None, tool_names: Sequence[str]
) -> ClaudeToolChoice:
    if tool_choice is None:
        return ClaudeToolChoice(type="auto")
    match tool_choice.mode:
        case "optional":
            return ClaudeToolChoice(type="auto")
        case "force_tool_use":
            return ClaudeToolChoice(type="any")
        case "force_specific_tool_use":
            if tool_choice.tool_name not in tool_names:
                raise ValueError(
                    f"tool_choice requires the tool {tool_choice.tool_name!r}, which is not one of "
                    f"the configured tools: {list(tool_names)}"
                )
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


# JSON schema keywords whose values are subschemas. Other keywords (e.g. default, examples, const,
# and enum) hold data and must not be modified.
_SUBSCHEMA_KEYWORDS = {
    "items",
    "additionalItems",
    "unevaluatedItems",
    "contains",
    "not",
    "if",
    "then",
    "else",
    "propertyNames",
}
_SUBSCHEMA_LIST_KEYWORDS = {"anyOf", "oneOf", "allOf", "prefixItems"}
_SUBSCHEMA_MAP_KEYWORDS = {
    "properties",
    "patternProperties",
    "dependentSchemas",
    "$defs",
    "definitions",
}


def _disallow_additional_properties(schema: dict[str, Any]) -> dict[str, Any]:
    """Sets additionalProperties to false on every object in a JSON schema.

    Claude structured output requires this on every object type, so objects that allow additional
    properties (e.g. dict fields) aren't supported and raise a ValueError.
    """

    def _subschema(value: Any) -> Any:  # noqa: ANN401
        if isinstance(value, dict):
            return _disallow_additional_properties(cast("dict[str, Any]", value))
        return value

    out = dict(schema)
    for key, value in schema.items():
        if key in _SUBSCHEMA_KEYWORDS:
            # "items" may also be a list of schemas in older drafts.
            out[key] = (
                [_subschema(v) for v in cast("list[Any]", value)]
                if isinstance(value, list)
                else _subschema(value)
            )
        elif key in _SUBSCHEMA_LIST_KEYWORDS and isinstance(value, list):
            out[key] = [_subschema(v) for v in cast("list[Any]", value)]
        elif key in _SUBSCHEMA_MAP_KEYWORDS and isinstance(value, dict):
            out[key] = {k: _subschema(v) for k, v in cast("dict[str, Any]", value).items()}

    schema_type = schema.get("type")
    is_object = schema_type == "object" or (
        isinstance(schema_type, list) and "object" in cast("list[Any]", schema_type)
    )
    if is_object:
        if schema.get("additionalProperties", False) is not False:
            raise ValueError(
                "Claude structured output requires additionalProperties to be false on every "
                "object, so objects with arbitrary keys (e.g. dict fields) aren't supported. "
                f"Schema: {json.dumps(schema)}"
            )
        out["additionalProperties"] = False
    return out


def validate_claude_sampling_params(model_id: str, config: LLMInferenceConfig) -> bool:
    """Returns true if temperature, top_p, and top_k should be sent to the model.

    Models that reject sampling params get none of them. The default temperature of 0 is dropped
    for these models so provider-neutral configs keep working, but any other temperature, top_p,
    or top_k raises a ValueError rather than being silently ignored.
    """
    capabilities = get_claude_model_capabilities(model_id)
    if capabilities is None or capabilities.sampling_params:
        return True
    if config.temperature != 0 or config.top_p is not None or config.top_k is not None:
        raise ValueError(
            f"{model_id} doesn't support temperature, top_p, or top_k. Leave them at their "
            "defaults (temperature=0, top_p=None, top_k=None)."
        )
    return False


# How many times to ask again when a model that doesn't support forced tool choice responds
# without calling a tool.
_FORCED_TOOL_CHOICE_RETRIES = 2


def _forced_tool_choice_instruction(model_id: str, config: LLMInferenceConfig) -> str | None:
    """Returns an instruction to call a tool, if the model can't be forced to call one.

    Returns None if the config doesn't force tool use or the model supports forced tool choice.
    """
    if (
        config.tools is None
        or config.tool_choice is None
        or _capabilities(model_id).forced_tool_choice
    ):
        return None
    match config.tool_choice.mode:
        case "optional":
            return None
        case "force_tool_use":
            return "You must call one of the tools in your response."
        case "force_specific_tool_use":
            return f"You must call the `{config.tool_choice.tool_name}` tool in your response."


class ClaudeRefusalError(Exception):
    """Exception raised when Claude refuses to answer."""

    stop_details: ClaudeStopDetails

    def __init__(self, stop_details: ClaudeStopDetails) -> None:  # noqa: D107
        self.stop_details = stop_details
        super().__init__(
            f"Claude refused to answer: {stop_details.category}, {stop_details.explanation}"
        )


class BedrockClaudeLLM(LLM):
    """Implements the LLM class for Bedrock Claude."""

    client: BedrockRuntimeClient

    def __init__(
        self,
        model_id: str = CLAUDE_4_5_HAIKU,
        client: BedrockRuntimeClient | None = None,
    ) -> None:
        """Initialize.

        Args:
            model_id: Model ID. Defaults to the model ID for Claude 4.5 Haiku.
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
        """Create a ClaudeInvokeLLMRequest from the given messages and inference config."""
        if config.json_mode:
            raise ValueError("JSON mode is not supported for Bedrock Claude.")
        if config.response_prefix is not None:
            raise ValueError("Response prefix is not supported for Bedrock Claude.")

        capabilities = _capabilities(self.model_id)
        send_sampling_params = validate_claude_sampling_params(self.model_id, config)

        tools = None
        tool_choice = None
        if config.tools is not None:
            tools = [_llm_tool_to_claude_tool(t) for t in config.tools]
            tool_choice = _llm_tool_choice_to_claude_tool_choice(
                config.tool_choice, [t.name for t in tools]
            )
            if not capabilities.forced_tool_choice:
                # The model rejects forced tool choice. An instruction is added to the system
                # prompt instead and prompt() asks again if the model doesn't call a tool.
                tool_choice = ClaudeToolChoice(type="auto")

            if capabilities.caching and config.cache_tools:
                # If the model supports caching we'll cache all of the tool definitions by default.
                # Every tool before the last tool will be included in this cache.
                last_tool = tools[-1]
                last_tool.cache_control = ClaudeCacheControl()

        system_content: list[ClaudeMessageContentType] | None = None

        if config.system_prompt:
            cache_system_prompt = capabilities.caching and config.cache_system_prompt
            cache_control = ClaudeCacheControl() if cache_system_prompt else None
            system_content = [
                ClaudeTextContent(cache_control=cache_control, text=config.system_prompt)
            ]
        tool_choice_instruction = _forced_tool_choice_instruction(self.model_id, config)
        if tool_choice_instruction is not None:
            # Added after the cached system prompt so it doesn't change the cached prefix.
            system_content = [
                *(system_content or []),
                ClaudeTextContent(text=tool_choice_instruction),
            ]

        output_config = None
        if config.structured_output_model is not None:
            output_config = ClaudeOutputConfig(
                format=JSONSchemaFormat(
                    schema=_disallow_additional_properties(
                        config.structured_output_model.model_json_schema()
                    )
                )
            )

        return ClaudeInvokeLLMRequest(
            max_tokens=config.max_tokens,
            system=system_content,
            temperature=config.temperature if send_sampling_params else None,
            top_k=config.top_k if send_sampling_params else None,
            top_p=config.top_p if send_sampling_params else None,
            tools=tools,
            tool_choice=tool_choice,
            messages=[_llm_message_to_claude_message(msg) for msg in messages],
            output_config=output_config,
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
        llm_msg = self._response_to_llm_message(response)

        instruction = _forced_tool_choice_instruction(self.model_id, inference_cfg)
        if instruction is not None:
            llm_msg = self._retry_until_tool_use(messages, inference_cfg, llm_msg, instruction)
        return llm_msg

    def _retry_until_tool_use(
        self,
        messages: Sequence[LLMMessage],
        inference_cfg: LLMInferenceConfig,
        llm_msg: LLMAssistantMessage,
        instruction: str,
    ) -> LLMAssistantMessage:
        """Asks the model again to call a tool when tool use was requested but couldn't be forced.

        Returns the last response, with token counts summed over all attempts. It may still lack
        a tool call if the model never made one.
        """

        def _needs_retry(msg: LLMAssistantMessage) -> bool:
            if msg.metadata is not None and msg.metadata.stop_reason == "max_tokens":
                # Asking again won't help if the model ran out of tokens.
                return False
            return not any(isinstance(c, LLMToolUseContent) for c in msg.content)

        input_tokens = llm_msg.metadata.input_tokens if llm_msg.metadata else 0
        output_tokens = llm_msg.metadata.output_tokens if llm_msg.metadata else 0
        retry_messages = list(messages)
        for _ in range(_FORCED_TOOL_CHOICE_RETRIES):
            if not _needs_retry(llm_msg):
                break
            log.warning(
                "%s responded without calling a tool even though tool use was required. "
                "Asking it again.",
                self.model_id,
            )
            if llm_msg.content:
                retry_messages.append(llm_msg)
            retry_messages.append(LLMUserMessage(content=instruction))
            response = self.invoke_model_with_request(
                self.create_request(retry_messages, inference_cfg)
            )
            llm_msg = self._response_to_llm_message(response)
            input_tokens += response.usage.input_tokens
            output_tokens += response.usage.output_tokens
        else:
            if _needs_retry(llm_msg):
                log.warning(
                    "%s didn't call a tool after %d retries.",
                    self.model_id,
                    _FORCED_TOOL_CHOICE_RETRIES,
                )

        if llm_msg.metadata is None:
            return llm_msg
        return llm_msg.model_copy(
            update={
                "metadata": llm_msg.metadata.model_copy(
                    update={"input_tokens": input_tokens, "output_tokens": output_tokens}
                )
            }
        )

    @timed_function
    def invoke_model_with_request(self, request: ClaudeInvokeLLMRequest) -> ClaudeResponse:
        """Invoke model with request and get a response back."""
        request_body = request.model_dump_json(exclude_none=True, by_alias=True)
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                # Regular Bedrock works without this value but LiteLLM requires it to work as proxy.
                contentType="application/json",
                body=request_body,
            )
        except botocore.exceptions.ClientError:
            log.exception("Request body: %s", request_body)
            raise
        response_body = response["body"].read().decode("UTF-8")
        try:
            claude_response = ClaudeResponse.model_validate_json(response_body)
        except ValidationError:
            log.exception("Failed to validate Claude response: %s", response_body)
            raise

        if claude_response.stop_details is not None:
            raise ClaudeRefusalError(stop_details=claude_response.stop_details)

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

    def _response_to_llm_message(self, response: ClaudeResponse) -> LLMAssistantMessage:
        def _to_llm_content(
            c: ClaudeTextContent
            | ClaudeToolUseContent
            | ClaudeThinkingContent
            | ClaudeRedactedThinkingContent,
        ) -> TextContent | LLMToolUseContent | ThinkingContent | RedactedThinkingContent:
            match c:
                case ClaudeTextContent(text=text):
                    return TextContent(text=text)
                case ClaudeToolUseContent():
                    return LLMToolUseContent(id=c.id, name=c.name, input=c.input)
                # Thinking blocks are kept because they must be sent back with tool results.
                case ClaudeThinkingContent():
                    return ThinkingContent(thinking=c.thinking, signature=c.signature)
                case ClaudeRedactedThinkingContent():
                    return RedactedThinkingContent(data=c.data)

        return LLMAssistantMessage(
            content=[_to_llm_content(c) for c in response.content],
            metadata=LLMResponseMetadata(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                stop_reason=response.stop_reason,
            ),
        )
