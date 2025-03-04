from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from e84_geoai_common.llm.models.converse.data_content_types import (
    ConverseDocumentContent,
    ConverseImageContent,
    ConverseJSONContent,
    ConverseTextContent,
    ConverseVideoContent,
)

# Converse uses camel case for its variables. Ignore any linting problems with this.
# ruff: noqa: N815


ConverseToolInputSchema = ConverseJSONContent


class ConverseToolSpec(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    name: str
    description: str | None = None
    inputSchema: ConverseToolInputSchema


class ConverseSingleTool(BaseModel):
    """Converse single tool model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    toolSpec: ConverseToolSpec


class Empty(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ConverseAnyToolChoice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    any: Empty = Empty()


class ConverseAutoToolChoice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    auto: Empty = Empty()


class ConverseSpecificTool(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str


class ConverseSpecificToolChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: ConverseSpecificTool


ConverseToolChoiceType = ConverseAnyToolChoice | ConverseAutoToolChoice | ConverseSpecificToolChoice


class ConverseToolConfig(BaseModel):
    """Converse tools model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    tools: Sequence[ConverseSingleTool]
    toolChoice: ConverseToolChoiceType


class ConverseToolUse(BaseModel):
    """Converse tool-use request model, inner."""

    model_config = ConfigDict(strict=True, extra="forbid")

    toolUseId: str
    name: str
    input: dict[str, Any]


class ConverseToolUseContent(BaseModel):
    """Converse tool-use request model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    toolUse: ConverseToolUse


ConverseToolResultContentType = (
    ConverseTextContent
    | ConverseJSONContent
    | ConverseDocumentContent
    | ConverseImageContent
    | ConverseVideoContent
)


class ConverseToolResult(BaseModel):
    """Converse tool inner result model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    toolUseId: str
    content: Sequence[ConverseToolResultContentType]
    status: Literal["success", "error"] | None = None


class ConverseToolResultContent(BaseModel):
    """Converse tool result modoel."""

    model_config = ConfigDict(strict=True, extra="forbid")

    toolResult: ConverseToolResult
