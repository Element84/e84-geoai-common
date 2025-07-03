from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from e84_geoai_common.llm.models.converse.msg_types import (
    ConverseAssistantMessage,
)

# Converse uses camel case for its variables. Ignore any linting problems with this.
# ruff: noqa: N815


class ConverseUsageInfo(BaseModel):
    """Usage info from the Converse API."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    inputTokens: int
    outputTokens: int
    totalTokens: int
    cacheReadInputTokens: int | None = Field(default=None)
    cacheWriteInputTokens: int | None = Field(default=None)


class ConverseMessageResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    message: ConverseAssistantMessage


class ConverseMetrics(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    latencyMs: int


class SystemContentBlock(BaseModel):
    """A system prompt block."""

    model_config = ConfigDict(strict=True, extra="forbid")

    text: str


class ConverseAdditionalModelRequestFields(BaseModel):
    """Converse additional fields for certain models."""

    model_config = ConfigDict(strict=True, extra="forbid")

    top_k: int | None


class ConverseResponse(BaseModel):
    """Converse response model."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    additionalModelResponseFields: dict[str, Any] | None = Field(default=None)
    metrics: ConverseMetrics
    output: ConverseMessageResponse
    performanceConfig: dict[str, Any] | None = Field(default=None)
    ResponseMetadata: dict[str, Any]
    role: Literal["assistant"] = "assistant"
    stopReason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    trace: dict[str, Any] | None = Field(default=None)
    usage: ConverseUsageInfo
