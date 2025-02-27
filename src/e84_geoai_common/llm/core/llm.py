from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Literal, Self

from function_schema.core import (  # type: ignore[reportMissingTypeStubs]
    get_function_schema,  # type: ignore[reportUnknownVariableType]
)
from pydantic import BaseModel, ConfigDict, Field

LLMMediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]


class TextContent(BaseModel):
    """Text context model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    text: str


class Base64ImageContent(BaseModel):
    """An image encoded for communication with an LLM."""

    model_config = ConfigDict(strict=True, extra="forbid")

    media_type: LLMMediaType
    data: str


class LLMMessage(BaseModel):
    """Standard representation of an LLM message.

    Specific LLM implementations should implement logic to translate to and
    from this representation.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    role: Literal["assistant", "user"] = "user"
    content: str | Sequence[TextContent | Base64ImageContent]

    def to_text_only(self) -> str:
        """Returns the message as text.

        Fails if the message is anything other than text
        """
        if isinstance(self.content, str):
            return self.content
        parts: list[str] = []
        for item in self.content:
            if not isinstance(item, TextContent):
                raise TypeError("The llm message is not just text")
            parts.append(item.text)
        return " ".join(parts)


class Tool(BaseModel):
    """Defines a tool that the model may use."""

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str
    description: str
    input_schema: dict[str, Any]

    @classmethod
    def from_function(cls, func: Callable[..., Any]) -> Self:
        """Construct from a Python funtion."""
        # This works because our tool class has the same fields as a Claude tool
        schema = get_function_schema(func, format="claude")  # type: ignore[reportUnknownVariableType]
        return cls.model_validate(schema)


class ToolUse(BaseModel):
    """Identifies a selected tool to use."""

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str
    input: dict[str, Any]


class LLMInferenceConfig(BaseModel):
    """Common inference options for LLMs.

    Specific LLM implementations should implement logic to translate these
    parameters to their respective APIs.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    system_prompt: str | None = Field(default=None, description="System Prompt")
    max_tokens: int = Field(default=1000, description="Maximum number of output tokens")
    temperature: float = Field(
        default=0,
        description="Temperature control for randomness. Closer to zero = more deterministic.",
    )
    top_p: float | None = Field(default=None, description="Top P for nucleus sampling.")
    top_k: int | None = Field(default=None, description="Top K for sampling")
    json_mode: bool = Field(
        default=False,
        description="If True, forces model to only outputs valid JSON.",
    )
    response_prefix: str | None = Field(
        default=None, description="Continue a pre-filled response instead of starting from scratch."
    )


class LLM(ABC):
    """An abstract base class for interacting with an LLM."""

    @abstractmethod
    def prompt(
        self,
        messages: Sequence[LLMMessage],
        inference_cfg: LLMInferenceConfig,
    ) -> LLMMessage:
        """Prompt the LLM with a message and optional conversation history.

        Returns the LLM message response.
        """

    def prompt_for_tools(
        self, messages: Sequence[LLMMessage], inference_cfg: LLMInferenceConfig, tools: list[Tool]
    ) -> ToolUse:
        """Prompts for a tool to use."""
        raise NotImplementedError
