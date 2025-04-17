from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

LLMMediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]


class TextContent(BaseModel):
    """Text content model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    text: str


class Base64ImageContent(BaseModel):
    """An image encoded for communication with an LLM."""

    model_config = ConfigDict(strict=True, extra="forbid")

    media_type: LLMMediaType
    data: str


class JSONContent(BaseModel):
    """JSON content model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    data: dict[str, Any]


class LLMToolUseContent(BaseModel):
    """Tool invocation request."""

    model_config = ConfigDict(strict=True, extra="forbid")

    id: str = Field(description="ID to track tool invocation and match it to tool output.")
    name: str = Field(description="Name of tool to invoke.")
    input: dict[str, Any] = Field(description="Inputs to invoke the tool with.")


LLMDataContentType = TextContent | JSONContent | Base64ImageContent


class LLMToolResultContent(BaseModel):
    """Tool invocation result."""

    model_config = ConfigDict(strict=True, extra="forbid")

    id: str = Field(description="ID of the corresponding LLMToolUseContent.")
    content: Sequence[LLMDataContentType] = Field(description="The result of the tool invocation.")
    status: Literal["success", "error"] | None = None


LLMMessageContentType = TextContent | Base64ImageContent | LLMToolUseContent | LLMToolResultContent


class LLMMessage(BaseModel):
    """Standard representation of an LLM message.

    Specific LLM implementations should implement logic to translate to and
    from this representation.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    role: Literal["assistant", "user"] = "user"
    content: str | Sequence[LLMMessageContentType]

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


class LLMTool(BaseModel):
    """Definition of a tool that an LLM may use."""

    model_config = ConfigDict(strict=True, extra="forbid")

    name: str
    description: str
    input_model: type[BaseModel] | None = Field(
        description="A Pydantic model describing the inputs to the tool. "
        "Can be set to None to indicate that the tool takes no inputs."
    )
    output_model: type[BaseModel] | None = Field(
        description="A Pydantic model describing the output of the tool. "
        "Can be set to None to indicate that the tool returns no text or JSON outputs."
    )
    execution_func: Callable[[LLMToolUseContent], LLMToolResultContent] | None = Field(
        default=None,
        exclude=True,
        description="A function that implements the execution of the tool. "
        "The function is expected to take in a LLMToolUseContent and return a "
        "LLMToolResultContent with a matching ID.",
    )

    def execute(self, tool_use_request: LLMToolUseContent) -> LLMToolResultContent:
        """Call execution_function with tool_use_request and return the result.

        Raises:
            NotImplementedError: If execution_func is not set.
            ValueError: If the ID of LLMToolResultContent returned by execution_func
                does not match the tool_use_request ID.
        """
        if self.execution_func is None:
            raise NotImplementedError("execution_func not set for this tool.")
        tool_result = self.execution_func(tool_use_request)
        if tool_result.id != tool_use_request.id:
            msg = (
                f"Tool result ID does not match tool call ID for tool {self.name} for input: "
                f"{tool_use_request.input}."
            )
            raise ValueError(msg)
        return tool_result


class LLMToolChoice(BaseModel):
    """Specification for constraining an LLM's choice of tools."""

    model_config = ConfigDict(strict=True, extra="forbid")

    mode: Literal["optional", "force_tool_use", "force_specific_tool_use"] = Field(
        default="optional",
        description="optional: the LLM may use any of the tools or not use a tool at all. "
        "force_tool_use: the LLM must use a tool but can choose which tool to use. "
        "force_specific_tool_use: the LLM must use a tool and it must be the tool specified in "
        "tool_name.",
    )
    tool_name: str | None = Field(
        None, description="Required if mode is 'force_specific_tool_use'."
    )


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
    tools: list[LLMTool] | None = Field(
        default=None, description="List of tools that the model may call."
    )
    tool_choice: LLMToolChoice | None = Field(
        default=None,
        description="Whether the model should use a specific "
        "tool, or any tool, or decide by itself.",
    )


class LLM(ABC):
    """An abstract base class for interacting with an LLM."""

    model_id: str

    @abstractmethod
    def prompt(
        self,
        messages: Sequence[LLMMessage],
        inference_cfg: LLMInferenceConfig,
    ) -> LLMMessage:
        """Prompt the LLM with a message and optional conversation history.

        Returns the LLM message response.
        """
