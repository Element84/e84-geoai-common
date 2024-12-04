import logging
from collections.abc import Callable, Sequence
from typing import Any, Literal, Self

import boto3
import botocore.exceptions
from function_schema.core import (  # type: ignore[reportMissingTypeStubs]
    get_function_schema,  # type: ignore[reportUnknownVariableType]
)
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from pydantic import BaseModel, ConfigDict, Field

from e84_geoai_common.llm.core.llm import LLM, LLMInferenceConfig, LLMMessage
from e84_geoai_common.util import timed_function

log = logging.getLogger(__name__)

# See https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
ANTHROPIC_API_VERSION = "bedrock-2023-05-31"
# https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html#model-ids-arns
CLAUDE_BEDROCK_MODEL_IDS = {
    "Claude 3 Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "Claude 3 Sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    "Claude 3 Opus": "anthropic.claude-3-opus-20240229-v1:0",
    "Claude Instant": "anthropic.claude-instant-v1",
    "Claude 3.5 Haiku": "anthropic.claude-3-5-haiku-20241022-v1:0",
    "Claude 3.5 Sonnet v2": "anthropic.claude-3-5-sonnet-20241022-v2:0",
}


class ClaudeTextContent(BaseModel):
    """Claude text context model."""

    type: Literal["text"] = "text"
    text: str

    def __str__(self) -> str:
        return self.text


class ClaudeToolUseContent(BaseModel):
    """Claude tool-use request model."""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ClaudeToolResultContent(BaseModel):
    """Claude tool result model."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str


class ClaudeMessage(LLMMessage):
    """Claude message base model."""

    role: Literal["assistant", "user"]
    content: (
        str
        | Sequence[
            ClaudeTextContent | ClaudeToolUseContent | ClaudeToolResultContent
        ]
    )

    @classmethod
    def from_llm_message(cls, message: LLMMessage) -> Self:
        """Construct from an LLMMessage."""
        return cls.model_validate(message.model_dump())


class ClaudeUserMessage(ClaudeMessage):
    """Claude user message model."""

    role: Literal["user"] = "user"
    content: str | Sequence[ClaudeTextContent | ClaudeToolResultContent]


class ClaudeAssistantMessage(ClaudeMessage):
    """Claude assistant message model."""

    role: Literal["assistant"] = "assistant"
    content: str | Sequence[ClaudeTextContent | ClaudeToolUseContent]


class ClaudeUsageInfo(BaseModel):
    """Claude usage-info model."""

    input_tokens: int
    output_tokens: int


class ClaudeResponse(BaseModel):
    """Claude response model."""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: Sequence[ClaudeTextContent | ClaudeToolUseContent]
    model: str
    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    stop_sequence: str | None
    usage: ClaudeUsageInfo

    def to_message(self) -> ClaudeMessage:
        """Convert to a ClaudeAssistantMessage."""
        return ClaudeAssistantMessage(role=self.role, content=self.content)


class ClaudeTool(BaseModel):
    """Representation of a tool that Claude can use."""

    name: str
    description: str
    input_schema: dict[str, Any]
    _func: Callable[..., Any]

    @classmethod
    def from_function(cls, func: Callable[..., Any]) -> Self:
        """Construct from a Python funtion."""
        schema = get_function_schema(func, format="claude")  # type: ignore[reportUnknownVariableType]
        out = cls.model_validate(schema)
        out._func = func  # noqa: SLF001
        return out

    def use(self, tool_use: ClaudeToolUseContent) -> ClaudeUserMessage:
        """Use tool and return the result as a ClaudeUserMessage."""
        func_out = self._func(**tool_use.input)
        result = ClaudeToolResultContent(
            tool_use_id=tool_use.id, content=str(func_out)
        )
        msg = ClaudeUserMessage(content=[result])
        return msg


class ClaudeToolChoice(BaseModel):
    """Claude tool choice model."""

    type: Literal["auto", "any", "tool"]
    name: str | None = None
    # disable_parallel_tool_use is documented in Anthropic docs but seems to
    # not be supported in Bedrock
    # disable_parallel_tool_use: bool | None = None  # noqa: ERA001


class ClaudeInvokeLLMRequest(BaseModel):
    """Represents a request to invoke Claude and get a response back."""

    model_config = ConfigDict(strict=True, extra="forbid")

    anthropic_version: str = ANTHROPIC_API_VERSION
    messages: list[ClaudeMessage] = Field(
        default_factory=list, description="List of LLM Messages"
    )
    system: str | None = Field(default=None, description="System Prompt")
    tools: list[ClaudeTool] | None = Field(
        default=None, description="List of tools that the model may call."
    )
    tool_choice: ClaudeToolChoice | None = Field(
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
        default=None, description="Top P for nucleus sampling"
    )
    top_k: int | None = Field(default=None, description="Top K for sampling")
    response_prefix: str | None = Field(
        default=None,
        description="Make Claude continue a pre-filled response instead of "
        'starting from sratch. Can be set to "{" to force "JSON mode".',
    )

    @classmethod
    def from_inference_config(
        cls,
        cfg: LLMInferenceConfig,
        messages: Sequence[ClaudeMessage] | None = None,
    ) -> Self:
        """Construct from an LLMInferenceConfig."""
        messages = [] if messages is None else list(messages)
        response_prefix = cfg.response_prefix
        if cfg.json_mode:
            if response_prefix is not None:
                msg = "response_prefix not supported with json_mode=True."
                raise ValueError(msg)
            response_prefix = "{"

        tools = None
        tool_choice = None
        if cfg.tools is not None:
            tools = [ClaudeTool.from_function(f) for f in cfg.tools]
            if cfg.tool_choice is None:
                tool_choice = ClaudeToolChoice(type="auto")
            elif cfg.tool_choice in ("auto", "any"):
                tool_choice = ClaudeToolChoice(type=cfg.tool_choice)
            else:
                tool_choice = ClaudeToolChoice(
                    type="tool", name=cfg.tool_choice
                )
            log.info(tool_choice)
        req = cls(
            messages=messages,
            system=cfg.system_prompt,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            response_prefix=response_prefix,
        )
        return req

    def to_request_body(self) -> str:
        """Convert to JSON request body."""
        if len(self.messages) == 0:
            msg = "Must specify at least one message."
            raise ValueError(msg)
        if self.response_prefix is not None:
            prefilled_response = ClaudeAssistantMessage(
                content=self.response_prefix
            )
            self.messages.append(prefilled_response)
        body = self.model_dump_json(
            exclude_none=True, exclude={"response_prefix"}
        )
        return body


class BedrockClaudeLLM(LLM):
    """Implements the LLM class for Bedrock Claude."""

    client: BedrockRuntimeClient

    def __init__(
        self,
        model_id: str = CLAUDE_BEDROCK_MODEL_IDS["Claude 3 Haiku"],
        client: BedrockRuntimeClient | None = None,
    ) -> None:
        """Initialize.

        Args:
            model_id: Model ID. Defaults to the model ID for Claude 3 Haiku.
            client: Optional pre-initialized boto3 client. Defaults to None.
        """
        self.model_id = model_id
        self.client = client or boto3.client("bedrock-runtime")  # type: ignore[reportUnknownMemberType]

    @timed_function
    def prompt(
        self,
        messages: Sequence[LLMMessage],
        inference_cfg: LLMInferenceConfig,
        *,
        auto_use_tools: bool = False,
    ) -> list[ClaudeMessage]:
        """Prompt the LLM with a message and optional conversation history."""
        if len(messages) == 0:
            msg = "Must specify at least one message."
            raise ValueError(msg)
        messages = [ClaudeMessage.from_llm_message(m) for m in messages]
        request = ClaudeInvokeLLMRequest.from_inference_config(
            inference_cfg, messages
        )
        response = self.invoke_model_with_request(request)
        if response.stop_reason == "tool_use" and auto_use_tools:
            assert request.tools is not None  # noqa: S101
            log.info("Tool-use requested:")
            log.info(response.content)
            tool_result_msgs = self.use_tools(response.content, request.tools)
            log.info("Tool-use results:")
            log.info(tool_result_msgs)
            new_messages = [
                *messages,
                response.to_message(),
                *tool_result_msgs,
            ]
            return self.prompt(
                new_messages,
                inference_cfg,
            )
        return [*messages, response.to_message()]

    @timed_function
    def invoke_model_with_request(
        self, request: ClaudeInvokeLLMRequest
    ) -> ClaudeResponse:
        """Invoke model with request and get a response back."""
        response_body = self._make_client_request(request)
        claude_response = self._parse_response(response_body, request)
        return claude_response

    def use_tools(
        self,
        content: Sequence[ClaudeTextContent | ClaudeToolUseContent],
        tools: list[ClaudeTool],
    ) -> list[ClaudeUserMessage]:
        """Fulfill all tool-use requests and return response messages."""
        tools_dict = {t.name: t for t in tools}
        out_messages: list[ClaudeUserMessage] = []
        for block in content:
            if not isinstance(block, ClaudeToolUseContent):
                continue
            tool = tools_dict[block.name]
            out_messages.append(tool.use(block))
        return out_messages

    def _parse_response(
        self, response_body: str, request: ClaudeInvokeLLMRequest
    ) -> ClaudeResponse:
        """Parse raw JSON response into a ClaudeResponse."""
        response = ClaudeResponse.model_validate_json(response_body)
        if request.response_prefix is not None:
            response = self._add_prefix_to_response(
                response, request.response_prefix
            )
        return response

    def _make_client_request(self, request: ClaudeInvokeLLMRequest) -> str:
        """Make model invocation request and return raw JSON response."""
        request_body = request.to_request_body()
        try:
            response = self.client.invoke_model(
                modelId=self.model_id, body=request_body
            )
        except botocore.exceptions.ClientError as e:
            log.error("Failed with %s", e)  # noqa: TRY400
            log.error("Request body: %s", request_body)  # noqa: TRY400
            raise
        response_body = response["body"].read().decode("UTF-8")
        return response_body

    def _add_prefix_to_response(
        self, response: ClaudeResponse, prefix: str
    ) -> ClaudeResponse:
        """Prepend prefix to the text of the first text-content block."""
        for content_block in response.content:
            if isinstance(content_block, ClaudeTextContent):
                content_block.text = prefix + content_block.text
                break
        return response
