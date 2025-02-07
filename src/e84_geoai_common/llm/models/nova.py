import logging
import json
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import boto3
import botocore.exceptions
from function_schema.core import (  # type: ignore[reportMissingTypeStubs]
    get_function_schema,  # type: ignore[reportUnknownVariableType]
)
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from pydantic import BaseModel, ConfigDict, Field

from e84_geoai_common.llm.core.llm import LLM, LLMInferenceConfig, LLMMessage
from e84_geoai_common.util import timed_function

if TYPE_CHECKING:
    from typing import Self

log = logging.getLogger(__name__)

# See https://docs.aws.amazon.com/nova/latest/userguide/using-converse-api.html
AMAZON_API_VERSION = "bedrock-2023-05-31"
# https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html#model-ids-arns
NOVA_BEDROCK_MODEL_IDS = {
    "Nova Canvas": "amazon.nova-canvas-v1:0",
    "Nova Lite": "amazon.nova-lite-v1:0",
    "Nova Micro": "amazon.nova-micro-v1:0",
    "Nova Pro": "amazon.nova-pro-v1:0",
    "Nova Reel": "amazon.nova-reel-v1:0",
}


class NovaTextContent(BaseModel):
    """Nova text context model."""

    # type: Literal["text"] = "text"
    text: str

    def __str__(self) -> str:
        return self.text

class NovaToolUseContent(BaseModel):
    """Nova tool-use request model."""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class NovaToolResultContent(BaseModel):
    """Nova tool result model."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str

class NovaMessage(LLMMessage):
    """Nova message base model."""

    role: Literal["assistant", "user"]
    content: (
        str
        | Sequence[
            NovaTextContent | NovaToolUseContent | NovaToolResultContent
        ]
    )

    @classmethod
    def from_llm_message(cls, message: LLMMessage) -> "Self":
        """Construct from an LLMMessage.""" 
        return cls(role=message.role, content=[NovaTextContent(text=message.content)])

class NovaUserMessage(NovaMessage):
    """Nova user message model."""

    role: Literal["user"] = "user"
    content: str | Sequence[NovaTextContent | NovaToolResultContent]


class NovaAssistantMessage(NovaMessage):
    """Nova assistant message model."""

    role: Literal["assistant"] = "assistant"
    content: str | Sequence[NovaTextContent | NovaToolResultContent]


class NovaUsageInfo(BaseModel):
    """Nova usage-info model."""

    input_tokens: int
    output_tokens: int

class NovaInferenceConfig(LLMInferenceConfig):
    max_new_tokens: int = Field(
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

class NovaResponseOutout(BaseModel):
    """Nova response output model."""

    message: NovaMessage

class NovaResponse(BaseModel):
    """Nova response model."""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: Sequence[NovaTextContent | NovaToolUseContent]
    model: str
    stop_reason: Literal["end_turn", "stop_sequence"]
    # stop_reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    stop_sequence: str | None
    usage: NovaUsageInfo

    def to_message(self) -> NovaMessage:
        """Convert to a NovaAssistantMessage."""
        return NovaAssistantMessage(role=self.role, content=self.content)

class NovaTool(BaseModel):
    """Representation of a tool that Nova can use."""

    name: str
    description: str
    input_schema: dict[str, Any]
    _func: Callable[..., Any]

    @classmethod
    def from_function(cls, func: Callable[..., Any]) -> "Self":
        """Construct from a Python funtion."""
        schema = get_function_schema(func, format="nova")  # type: ignore[reportUnknownVariableType]
        out = cls.model_validate(schema)
        out._func = func  # noqa: SLF001
        return out

    def use(self, tool_use: NovaToolUseContent) -> NovaUserMessage:
        """Use tool and return the result as a NovaUserMessage."""
        func_out = self._func(**tool_use.input)
        result = NovaToolResultContent(
            tool_use_id=tool_use.id, content=str(func_out)
        )
        msg = NovaUserMessage(content=[result])
        return msg

class NovaToolChoice(BaseModel):
    """Nova tool choice model."""

    type: Literal["auto", "any", "tool"]
    name: str | None = None
    # disable_parallel_tool_use is documented in Anthropic docs but seems to
    # not be supported in Bedrock
    # disable_parallel_tool_use: bool | None = None  # noqa: ERA001
class NovaInvokeLLMRequest(BaseModel):
    """Represents a request to invoke Nova and get a response back."""

    schemaVersion: str = Field(
        default="messages-v1", description="Version of the schema"
    )
    messages: list[NovaMessage] = Field(
        default_factory=list, description="List of LLM Messages"
    )
    # system: str | None = Field(default=None, description="System Prompt")
    # system: list[NovaTextContent] = Field(
    #     default=None, description="System Prompt"
    # )
    inferenceCfg: NovaInferenceConfig | None = Field(
        default=None, description="Inference Config"
    )
    response_prefix: str | None = Field(
        default=None,
        description="Make Nova continue a pre-filled response instead of "
        'starting from sratch. Can be set to "{" to force "JSON mode".',
    )
    # model_config = ConfigDict(strict=True, extra="forbid")
    # amazon_version: str = AMAZON_API_VERSION

    tools: list[NovaTool] | None = Field(
        default=None, description="List of tools that the model may call."
    )
    tool_choice: NovaToolChoice | None = Field(
        default=None,
        description="Whether the model should use a specific "
        "tool, or any tool, or decide by itself.",
    )

    @classmethod
    def from_inference_config(
        cls,
        cfg: LLMInferenceConfig,
        messages: Sequence[NovaMessage] | None = None,
    ) -> "Self":
        """Construct from an NovaInferenceConfig."""
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
            tools = [NovaTool.from_function(f) for f in cfg.tools]
            if cfg.tool_choice is None:
                tool_choice = NovaToolChoice(type="auto")
            elif cfg.tool_choice in ("auto", "any"):
                tool_choice = NovaToolChoice(type=cfg.tool_choice)
            else:
                tool_choice = NovaToolChoice(
                    type="tool", name=cfg.tool_choice
                )
            log.info(tool_choice)
        req = cls(
            messages=messages,
            system=[{"text": cfg.system_prompt}],
            inferenceConfig=cfg,
            response_prefix=response_prefix,
        )
        return req

    def to_request_body(self) -> str:
        """Convert to JSON request body."""
        if len(self.messages) == 0:
            msg = "Must specify at least one message."
            raise ValueError(msg)
        if self.response_prefix is not None:
            prefilled_response = NovaAssistantMessage(
                content=[NovaTextContent(text=self.response_prefix)]
            )
            self.messages.append(prefilled_response)
        body = self.model_dump_json(
            exclude_none=True, exclude={"response_prefix"}
        )
        return body


class BedrockNovaLLM(LLM):
    """Implements the LLM class for Bedrock Nova."""

    client: BedrockRuntimeClient

    def __init__(
        self,
        model_id: str = NOVA_BEDROCK_MODEL_IDS["Nova Pro"],
        client: BedrockRuntimeClient | None = None,
    ) -> None:
        """Initialize.

        Args:
            model_id: Model ID. Defaults to the model ID for Nova Pro.
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
    ) -> list[NovaMessage]:
        """Prompt the LLM with a message and optional conversation history."""
        if len(messages) == 0:
            msg = "Must specify at least one message."
            raise ValueError(msg)
        messages = [NovaMessage.from_llm_message(m) for m in messages]
        request = NovaInvokeLLMRequest.from_inference_config(
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
        self, request: NovaInvokeLLMRequest
    ) -> NovaResponse:
        """Invoke model with request and get a response back."""
        response_body = self._make_client_request(request)
        print("invoke_model_with_request#response_body", response_body)
        nova_response = self._parse_response(response_body, request)
        return nova_response

    def use_tools(
        self,
        content: Sequence[NovaTextContent | NovaToolUseContent],
        tools: list[NovaTool],
    ) -> list[NovaUserMessage]:
        """Fulfill all tool-use requests and return response messages."""
        tools_dict = {t.name: t for t in tools}
        out_messages: list[NovaUserMessage] = []
        for block in content:
            if not isinstance(block, NovaToolUseContent):
                continue
            tool = tools_dict[block.name]
            out_messages.append(tool.use(block))
        return out_messages

    def _parse_response(
        self, response_body: str, request: NovaInvokeLLMRequest
    ) -> NovaResponse:
        """Parse raw JSON response into a NovaResponse."""
        # json_string = json.dumps(response_json, indent=2)
        # print("------response_body", response_body)

        # response_json = {
        #     "id": response_body.get("id", ""),
        #     "type": response_body.get("type", "message"),
        #     "role": response_body.get("role", "assistant"),
        #     "content": [
        #         {"text": block.get("text", "")[7:-3].strip()}
        #         for block in response_body.get("output", {}).get("message", {}).get("content", [])
        #     ],
        #     "model": response_body.get("model", ""),
        #     "stop_reason": response_body.get("stopReason", "end_turn"),
        #     "stop_sequence": response_body.get("stopSequence"),
        #     "usage": {
        #         "input_tokens": response_body["usage"]["inputTokens"],
        #         "output_tokens": response_body["usage"]["outputTokens"]
        #     },
        # }

        # # Convert to JSON string if needed
        # json_string = json.dumps(response_json, indent=2)
        print("_parse_response#response_body", response_body.get("output", {}).get("message", {}).get("content", []))

        response = NovaResponse.model_validate_json(response_body)
        if request.response_prefix is not None:
            response = self._add_prefix_to_response(
                response, request.response_prefix
            )
        return response
    
        # Convert JSON string to a dictionary
        response_data = response_body
        
        # Handle nested structure
        content_list = response_data.get("output", {}).get("message", {}).get("content", [])
        if not content_list:
            raise ValueError("No content in response body: %s" % response_body)


        # Construct NovaResponse with extracted data
        response = NovaResponse(
            id=response_data.get("id", ""),
            type=response_data.get("type", "message"),
            role=response_data.get("role", "assistant"),
            # content=[NovaTextContent(text=block["text"]) for block in content_list],
            content=[NovaTextContent(text=response_data.get("output", {}).get("message", {}).get("content", [])[0].get("text", ""))],
            model=response_data.get("model", ""),
            stop_reason=response_data.get("stopReason", "end_turn"),
            stop_sequence=response_data.get("stopSequence"),
            usage=NovaUsageInfo(
                input_tokens=response_data["usage"]["inputTokens"],
                output_tokens=response_data["usage"]["outputTokens"]
            ),
        )

        return response

    def _make_client_request(self, request: NovaInvokeLLMRequest) -> str:
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
        print("******", response_body)
        return response_body


        """Make model invocation request and return raw JSON response."""
        request_body = request.to_request_body()
        try:
            response = self.client.invoke_model(
                modelId=self.model_id, body=request_body
            )
            response_body = response["body"].read().decode("UTF-8")
            log.debug("Raw response body: %s", response_body)
            # Parse JSON to extract the actual message content
            # return json.loads(response_body)  # Import json if not already done
            if "output" in response_json and "message" in response_json["output"]:
                return json.dumps(response_json["output"]["message"])  # Return only the message
            else:
                raise ValueError("Unexpected response structure: %s" % response_body)
        except botocore.exceptions.ClientError as e:
            log.error("Failed with %s", e)
            log.error("Request body: %s", request_body)
            raise
        return response_body


    def _add_prefix_to_response(
        self, response: NovaResponse, prefix: str
    ) -> NovaResponse:
        """Prepend prefix to the text of the first text-content block."""
        for content_block in response.content:
            if isinstance(content_block, NovaTextContent):
                content_block.text = prefix + content_block.text
                break
        return response
