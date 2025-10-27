from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Literal, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class CachePointContent(BaseModel):
    """A marker within a list of content that indicates everything before this can be cached."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["default"] = "default"


LLMMessageContentType = (
    TextContent | Base64ImageContent | LLMToolUseContent | LLMToolResultContent | CachePointContent
)


class LLMResponseMetadata(BaseModel):
    """Metadata associated with the message."""

    model_config = ConfigDict(strict=True, extra="forbid")

    input_tokens: int
    output_tokens: int
    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]


class LLMMessage(BaseModel, frozen=True):
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


class LLMAssistantMessage(LLMMessage, frozen=True):
    """Message form LLM."""

    model_config = ConfigDict(strict=True, extra="forbid")

    role: Literal["assistant"] = "assistant"
    metadata: LLMResponseMetadata | None = None


class LLMUserMessage(LLMMessage, frozen=True):
    """Message to LLM."""

    model_config = ConfigDict(strict=True, extra="forbid")

    role: Literal["user"] = "user"


class LLMTool(BaseModel, frozen=True):
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


LLMToolContext = TypeVar("LLMToolContext")

ExecutableFunction = Callable[[LLMToolContext, LLMToolUseContent], LLMToolResultContent]


class ExecutableLLMTool[LLMToolContext](ABC):
    """The executable version of a tool."""

    tool_spec: LLMTool

    def __init__(self, tool_spec: LLMTool) -> None:
        """Constructor."""
        self.tool_spec = tool_spec

    def execute(
        self, context: LLMToolContext, tool_use_request: LLMToolUseContent
    ) -> LLMToolResultContent:
        """Call execution_function with tool_use_request and return the result.

        Args:
            context: The context to pass to the execution function.
            tool_use_request: The tool invocation request.

        Raises:
            NotImplementedError: If execution_func is not set.
            ValueError: If the ID of LLMToolResultContent returned by execution_func
                does not match the tool_use_request ID.
        """
        tool_result = self._execute(context, tool_use_request)
        self._validate_tool_result_id(tool_result, tool_use_request)
        return tool_result

    def _validate_tool_result_id(
        self, tool_result: LLMToolResultContent, tool_use_request: LLMToolUseContent
    ) -> None:
        if tool_result.id != tool_use_request.id:
            msg = (
                f"Tool result ID '{tool_result.id}' does not match "
                f"tool call ID '{tool_use_request.id}'."
                f"Tool call:\n{tool_use_request}\n"
                f"Tool result:\n{tool_result}\n"
            )
            raise ValueError(msg)

    @abstractmethod
    def _execute(
        self,
        context: LLMToolContext,
        tool_use_request: LLMToolUseContent,
    ) -> LLMToolResultContent:
        """The function that executes the tool.

        This function should be overridden by subclasses to provide the actual
        implementation of the tool.
        """


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
    cache_system_prompt: bool = Field(
        default=True,
        description="Indicates if the system prompt should be marked as cacheable in the LLM.",
    )
    max_tokens: int = Field(default=1000, description="Maximum number of output tokens")
    temperature: float = Field(
        default=0,
        description="Temperature control for randomness. Closer to zero = more deterministic.",
    )
    top_p: float | None = Field(default=None, description="Top P for nucleus sampling.")
    top_k: int | None = Field(default=None, description="Top K for sampling")
    json_mode: bool = Field(
        default=False,
        description="If True, forces the model to only output valid JSON. "
        "This is useful for generating structured output. "
        "Note: this currently forces the response to begin as a JSON string "
        "and ends the response as soon as the JSON string ends. As such, this "
        "is not compatible with tools. In order to get both structured output "
        "and tool use, consider other mechanisms such as XML tags or limiting "
        "responses to tool calls exclusively.",
    )
    response_prefix: str | None = Field(
        default=None, description="Continue a pre-filled response instead of starting from scratch."
    )
    tools: list[LLMTool] | None = Field(
        default=None, description="List of tools that the model may call."
    )
    cache_tools: bool = Field(
        default=True,
        description="Indicates if the tools should be marked as cacheable in the LLM.",
    )
    tool_choice: LLMToolChoice | None = Field(
        default=None,
        description="Whether the model should use a specific "
        "tool, or any tool, or decide by itself.",
    )

    @model_validator(mode="after")
    def _disallow_tools_with_json_mode(self) -> Self:
        if self.json_mode and self.tools:
            raise ValueError("json_mode is not supported with tools.")
        return self


class LLM(ABC):
    """An abstract base class for interacting with an LLM."""

    model_id: str

    @abstractmethod
    def prompt(
        self,
        messages: Sequence[LLMMessage],
        inference_cfg: LLMInferenceConfig,
    ) -> LLMAssistantMessage:
        """Prompt the LLM with a message and optional conversation history.

        Returns the LLM message response.
        """

    # @abstractmethod
    # def create_request(self, messages: Sequence[LLMMessage], config: LLMInferenceConfig) -> Any:
    #     """Create the LLM Request payload given a message and optional conversation history."""
