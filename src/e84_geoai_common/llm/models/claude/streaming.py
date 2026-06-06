"""Pydantic models for Claude streaming response events.

These models represent the server-sent events returned by Bedrock's
invoke_model_with_response_stream API when using Anthropic Claude models.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Delta types (content within content_block_delta events)
# =============================================================================


class ClaudeTextDelta(BaseModel, frozen=True):
    """A text delta within a streaming content block."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["text_delta"] = "text_delta"
    text: str


class ClaudeInputJsonDelta(BaseModel, frozen=True):
    """A partial JSON delta for tool use input."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["input_json_delta"] = "input_json_delta"
    partial_json: str


ClaudeStreamDelta = Annotated[
    ClaudeTextDelta | ClaudeInputJsonDelta,
    Field(discriminator="type"),
]


# =============================================================================
# Content block types (within content_block_start events)
# =============================================================================


class ClaudeStreamTextBlock(BaseModel, frozen=True):
    """A text content block being started."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["text"] = "text"
    text: str


class ClaudeStreamToolUseBlock(BaseModel, frozen=True):
    """A tool_use content block being started."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


ClaudeStreamContentBlock = Annotated[
    ClaudeStreamTextBlock | ClaudeStreamToolUseBlock,
    Field(discriminator="type"),
]


# =============================================================================
# Message-level metadata types
# =============================================================================


class ClaudeStreamMessageUsage(BaseModel, frozen=True):
    """Usage info from a message_start event."""

    model_config = ConfigDict(strict=True, extra="forbid")

    input_tokens: int
    output_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class ClaudeStreamMessageInfo(BaseModel, frozen=True):
    """Message metadata from a message_start event."""

    model_config = ConfigDict(strict=True, extra="forbid")

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: ClaudeStreamMessageUsage


class ClaudeStreamMessageDeltaInfo(BaseModel, frozen=True):
    """Delta info from a message_delta event."""

    model_config = ConfigDict(strict=True, extra="forbid")

    stop_reason: Literal[
        "end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"
    ]
    stop_sequence: str | None = None


class ClaudeStreamMessageDeltaUsage(BaseModel, frozen=True):
    """Usage info from a message_delta event (cumulative output tokens)."""

    model_config = ConfigDict(strict=True, extra="forbid")

    output_tokens: int


# =============================================================================
# Streaming event types
# =============================================================================


class ClaudeStreamMessageStart(BaseModel, frozen=True):
    """The initial event in a stream, containing message metadata."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["message_start"] = "message_start"
    message: ClaudeStreamMessageInfo


class ClaudeStreamContentBlockStart(BaseModel, frozen=True):
    """Signals the start of a new content block."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: ClaudeStreamContentBlock


class ClaudeStreamContentBlockDelta(BaseModel, frozen=True):
    """A delta update within a content block."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: ClaudeStreamDelta


class ClaudeStreamContentBlockStop(BaseModel, frozen=True):
    """Signals the end of a content block."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class ClaudeStreamMessageDelta(BaseModel, frozen=True):
    """A message-level delta, typically containing stop_reason and final usage."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["message_delta"] = "message_delta"
    delta: ClaudeStreamMessageDeltaInfo
    usage: ClaudeStreamMessageDeltaUsage


class ClaudeStreamMessageStop(BaseModel, frozen=True):
    """The final event in a stream."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["message_stop"] = "message_stop"


ClaudeStreamEvent = Annotated[
    ClaudeStreamMessageStart
    | ClaudeStreamContentBlockStart
    | ClaudeStreamContentBlockDelta
    | ClaudeStreamContentBlockStop
    | ClaudeStreamMessageDelta
    | ClaudeStreamMessageStop,
    Field(discriminator="type"),
]


# =============================================================================
# Parsing
# =============================================================================

# Event types that we silently skip (e.g. ping)
_SKIPPABLE_EVENT_TYPES = frozenset({"ping"})


class ClaudeStreamError(Exception):
    """Raised when a stream error event is received."""

    def __init__(self, error_data: dict[str, Any]) -> None:  # noqa: D107
        self.error_data = error_data
        super().__init__(f"Stream error: {error_data}")


_EVENT_TYPE_MAP: dict[str, type[BaseModel]] = {
    "message_start": ClaudeStreamMessageStart,
    "content_block_start": ClaudeStreamContentBlockStart,
    "content_block_delta": ClaudeStreamContentBlockDelta,
    "content_block_stop": ClaudeStreamContentBlockStop,
    "message_delta": ClaudeStreamMessageDelta,
    "message_stop": ClaudeStreamMessageStop,
}


def parse_stream_event(data: dict[str, Any]) -> ClaudeStreamEvent | None:
    """Parse a raw streaming event dict into a typed ClaudeStreamEvent.

    Args:
        data: The raw JSON dict decoded from a streaming chunk.

    Returns:
        A typed ClaudeStreamEvent, or None if the event should be skipped (e.g. ping).

    Raises:
        ClaudeStreamError: If the event is an error event.
        ValueError: If the event type is unrecognized.
    """
    event_type = data.get("type")

    if event_type in _SKIPPABLE_EVENT_TYPES:
        return None

    if event_type == "error":
        raise ClaudeStreamError(data.get("error", data))

    model_cls = _EVENT_TYPE_MAP.get(event_type)  # type: ignore[arg-type]
    if model_cls is None:
        raise ValueError(f"Unrecognized stream event type: {event_type!r}")

    return model_cls.model_validate(data)  # type: ignore[return-value]
