from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from e84_geoai_common.llm.models.converse.data_content_types import (
    ConverseImageContent,
    ConverseTextContent,
)
from e84_geoai_common.llm.models.converse.tool_use_types import (
    ConverseToolResultContent,
    ConverseToolUseContent,
)

# Converse uses camel case for its variables. Ignore any linting problems with this.
# ruff: noqa: N815


ConverseMessageContentType = (
    ConverseTextContent | ConverseImageContent | ConverseToolUseContent | ConverseToolResultContent
)


class ConverseMessage(BaseModel):
    """Converse base model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    role: Literal["assistant", "user"]
    content: Sequence[ConverseMessageContentType]


class ConverseUserMessage(BaseModel):
    """Converse user message model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    role: Literal["user"] = "user"
    content: list[ConverseTextContent | ConverseToolResultContent]


class ConverseAssistantMessage(BaseModel):
    """Converse assistant message model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    role: Literal["assistant"] = "assistant"
    content: list[ConverseTextContent | ConverseToolUseContent]
