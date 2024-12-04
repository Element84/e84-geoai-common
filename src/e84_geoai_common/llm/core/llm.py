from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    """Standard representation of an LLM message.

    Specific LLM implementations should implement logic to translate to and
    from this representation.
    """

    role: str
    content: str | Sequence[Any]


class LLMInferenceConfig(BaseModel):
    """Common inference options for LLMs.

    Specific LLM implementations should implement logic to translate these
    parameters to their respective APIs.
    """

    system_prompt: str | None = Field(
        default=None, description="System Prompt"
    )
    tools: list[Callable[..., Any]] | None = Field(
        default=None, description="List of tools that the model may call."
    )
    tool_choice: str | None = Field(
        default=None,
        description="Whether the model should use a specific "
        "tool, or any tool, or decide by itself.",
    )
    max_tokens: int = Field(
        default=1000, description="Maximum number of output tokens"
    )
    temperature: float = Field(
        default=0,
        description="Temperature control for randomness. "
        "Closer to zero = more deterministic.",
    )
    top_p: float | None = Field(
        default=None, description="Top P for nucleus sampling."
    )
    top_k: int | None = Field(default=None, description="Top K for sampling")
    json_mode: bool = Field(
        default=False,
        description="If True, forces model to only outputs valid JSON.",
    )
    response_prefix: str | None = Field(
        default=None,
        description="Continue a pre-filled response instead of "
        "starting from sratch.",
    )


class LLM(ABC):
    """An abstract base class for interacting with an LLM."""

    @abstractmethod
    def prompt(
        self,
        messages: Sequence[LLMMessage],
        inference_cfg: LLMInferenceConfig,
        *,
        auto_use_tools: bool = False,
    ) -> Sequence[LLMMessage]:
        """Prompt the LLM with a message and optional conversation history."""
