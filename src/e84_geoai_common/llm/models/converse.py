import json
import logging
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


CONVERSE_BEDROCK_MODEL_IDS = {
    "Claude 3 Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "Claude 3 Sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    "Claude 3 Opus": "anthropic.claude-3-opus-20240229-v1:0",
    "Claude Instant": "anthropic.claude-instant-v1",
    "Claude 3.5 Haiku": "anthropic.claude-3-5-haiku-20241022-v1:0",
    "Claude 3.5 Sonnet v2": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    # "Nova Canvas": "amazon.nova-canvas-v1:0",
    "Nova Lite": "us.amazon.nova-lite-v1:0", #added 'us' infront for invocation
    "Nova Micro": "us.amazon.nova-micro-v1:0",
    "Nova Pro": "us.amazon.nova-pro-v1:0",
    # "Nova Reel": "amazon.nova-reel-v1:0",
    "Llama 3.1 70B Instruct": "us.meta.llama3-1-70b-instruct-v1:0",
    "Llama 3.1 8B Instruct": "us.meta.llama3-1-8b-instruct-v1:0",
    "Llama 3.2 11B Vision Instruct": "us.meta.llama3-2-11b-instruct-v1:0",
    "Llama 3.2 1B Instruct": "us.meta.llama3-2-1b-instruct-v1:0",
    "Llama 3.2 3B Instruct": "us.meta.llama3-2-3b-instruct-v1:0",
    "Llama 3.2 90B Vision Instruct": "us.meta.llama3-2-90b-instruct-v1:0",
    "Llama 3.3 70B Instruct": "us.meta.llama3-3-70b-instruct-v1:0"
}


class ConverseTextContent(BaseModel):
    """Converse text context model."""
    text: str

    def __str__(self) -> str:
        return self.text


class ConverseToolUseInnerContent(BaseModel):
    """Converse tool-use request model, inner."""

    toolUseId: str  # noqa: N815
    name: str
    input: dict[str, Any]


class ConverseToolUseContent(BaseModel):
    """Converse tool-use request model."""
    toolUse: ConverseToolUseInnerContent # noqa: N815


class ConverseToolResultInnerContent(BaseModel):
    """Converse tool inner result model."""
    toolUseId: str # noqa: N815
    content: list[dict[str, Any]]
    status: str | None = None


class ConverseToolResultContent(BaseModel):
    """Converse tool result modoel."""
    toolResult: ConverseToolResultInnerContent # noqa: N815


class ConverseMessage(LLMMessage):
    """Converse base model."""

    role: Literal["assistant", "user"]
    content: (
        str
        | Sequence[
            ConverseTextContent | ConverseToolUseContent |
            ConverseToolResultContent
        ]
    )

    @classmethod
    def from_llm_message(cls, message: LLMMessage) -> "Self":
        """Construct from an LLMMessage."""
        data = message.model_dump()
        #this will not work for images and other files
        if isinstance(data["content"], str):
            data["content"] = [{"text": data["content"]}]
        return cls.model_validate(data)


class ConverseUserMessage(ConverseMessage):
    """Converse user message model."""
    role: Literal["user"] = "user"
    content: list[ConverseTextContent | ConverseToolResultContent]


class ConverseAssistantMessage(ConverseMessage):
    """Converse assistant message model."""
    role: Literal["assistant"] = "assistant"
    content: list[ConverseTextContent | ConverseToolUseContent]


class ConverseUsageInfo(BaseModel):
    """Usage info from the Converse API."""
    inputTokens: int # noqa: N815
    outputTokens: int # noqa: N815
    totalTokens: int # noqa: N815


class ConverseResponse(BaseModel):
    """Converse response model."""
    role: Literal["assistant"] = "assistant"
    content: list[ConverseTextContent | ConverseToolUseContent]
    stop_reason: Literal[
        "end_turn", "max_tokens", "stop_sequence", "tool_use"
    ] = Field(alias="stopReason")
    usage: ConverseUsageInfo

    def to_message(self) -> ConverseMessage:
        """Convert to a ConverseAssistantMessage."""
        return ConverseAssistantMessage(role=self.role, content=self.content)


class ConverseToolSpec(BaseModel):
    """Representation of a tool that Converse can use."""
    name: str
    description: str
    inputSchema: dict[str, Any]  # noqa: N815
    _func: Callable[..., Any]

    @classmethod
    def from_function(cls, func: Callable[..., Any]) -> "Self":
        """Construct from a python function."""
        schema = get_function_schema(func)
        if "parameters" in schema:
            #Edits openAI's schema to fit converse
            input_sch = {"json": schema.pop("parameters")}
            schema["inputSchema"] = input_sch # type: ignore  # noqa: PGH003
        out = cls.model_validate(schema)
        out._func = func  # noqa: SLF001
        return out

    def use(self, tool_use: ConverseToolUseContent) -> ConverseUserMessage:
        """Use tool and return result as a ConverseUserMessage."""
        try:
            func_out = self._func(**tool_use.toolUse.input)
            result_content = {"json": {"result": str(func_out)}}
            status = None
        except Exception as ex:  # noqa: BLE001
            result_content = {"text": str(ex)}
            status = "error"
        block = ConverseToolResultContent(
            toolResult=ConverseToolResultInnerContent(
                toolUseId=tool_use.toolUse.toolUseId,
                content= [result_content],
                status = status
            )
        )
        return ConverseUserMessage(content=[block])


class ConverseSingleTool(BaseModel):
    """Converse single tool model."""

    toolSpec: ConverseToolSpec  # noqa: N815


class ConverseTools(BaseModel):
    """Converse tools model."""

    tools: Sequence[ConverseSingleTool]

class ConverseToolChoice(BaseModel):
    """Converse tool choice model."""

    #only anthropic and mistral models use tool choice. It is not implemented
    #in the code here yet - this is a dummy class.
    type: Literal["auto", "any", "tool"]
    name: str | None = None
    # disable_parallel_tool_use is documented in Anthropic docs but seems to
    # not be supported in Bedrock
    # disable_parallel_tool_use: bool | None = None  # noqa: ERA001

class SystemContentBlock(BaseModel):
    """A system prompt block."""
    text: str

class ConverseInferenceConfig(BaseModel):
    """Converse inference config model."""

    maxTokens: int | None  # noqa: N815
    stopSequences: Sequence[str] | None  # noqa: N815
    temperature: float | None
    topP: float | None  # noqa: N815

class ConverseAdditionalModelRequestFields(BaseModel):
    """Converse additional fields for certain models."""

    top_k: int | None


class ConverseRequest(BaseModel):
    """Converse request model."""

    modelId: str  # noqa: N815
    messages: Sequence[ConverseMessage]
    toolConfig: ConverseTools | None  # noqa: N815
    system: Sequence[SystemContentBlock] | None
    inferenceConfig: ConverseInferenceConfig | None  # noqa: N815
    additionalModelRequestFields: ConverseAdditionalModelRequestFields | None  # noqa: N815


class ConverseInvokeLLMRequest(BaseModel):
    """Represents a request to invoke Converse API and get a response back."""
    model_config = ConfigDict(strict=True, extra="forbid")

    messages: list[ConverseMessage] = Field(
        default_factory=list,
        description="List of conversation messages",
    )
    system: list[SystemContentBlock] | None = Field(
        default=None,
        description="Optional system prompt(s)",
    )
    tools: ConverseTools | None = Field(
        default=None,
        description="Tools for the model to use",
    )
    tool_choice: ConverseToolChoice | None = Field(
        default=None,
        description="Tool choice (specific, auto, or any)",
    )
    model_id: str = Field(
        default= "us.anthropic.claude-3-haiku-20240307-v1:0",
        description="Model id for the LLM"
    )
    max_tokens: int = Field(
        default=1000,
        description="Max tokens for the response",
    )
    temperature: float = Field(
        default=0,
        description="Temperature for randomness",
    )
    top_p: float | None = Field(
        default=None,
        description="Top-p sampling value",
    )
    top_k: int | None = Field(
        default=None,
        description="Top-k sampling value",
    )
    response_prefix: str | None = Field(
        default=None,
        description="Prefill prefix for the model response",
    )

    @classmethod
    def from_inference_config(
        cls,
        cfg: LLMInferenceConfig,
        model_id: str,
        messages: Sequence[ConverseMessage] | None = None,
    ) -> "Self":
        """Construct from an LLMInferenceConfig."""
        messages = [] if messages is None else list(messages)
        response_prefix = cfg.response_prefix
        if cfg.json_mode:
            if response_prefix is not None:
                msg = "response_prefix not supported with json_mode=True."
                raise ValueError(msg)
            response_prefix = "{"

        sys_prompts = None
        if cfg.system_prompt:
            #For now, just support for a single system prompt content block
            sys_prompts = [SystemContentBlock(text=cfg.system_prompt)]

        tools = None
        tool_choice = None
        if cfg.tools is not None:
            tools = ConverseTools(
                tools = [
                    ConverseSingleTool(toolSpec=ConverseToolSpec.from_function(f))
                    for f in cfg.tools
                ],
            )
        # tool choice not implemented yet. It is only partially used in
        # antrophic and mistral models. It seems to be automatic by default
        # on all models that can use tools
        return cls(
            model_id=model_id,
            messages=messages,
            system=sys_prompts,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            response_prefix=cfg.response_prefix,
        )

    def to_request_body(self) -> str:
        """Convert to JSON request body."""
        if not self.messages:
            msg = "Must specify at least one message."
            raise ValueError(msg)
        if self.response_prefix is not None:
            prefix_msg = ConverseAssistantMessage(
                content=[ConverseTextContent(text=self.response_prefix)],
            )
            self.messages.append(prefix_msg)

        inference_config = ConverseInferenceConfig(
            maxTokens = self.max_tokens,
            stopSequences=None, #stop sequence not implemented yet
            temperature=self.temperature,
            topP=self.top_p
        )

        additional_model_request_fields = ConverseAdditionalModelRequestFields(
            top_k=self.top_k
        )
        if not additional_model_request_fields.model_dump(exclude_none=True):
            additional_model_request_fields = None

        request = ConverseRequest(
            modelId=self.model_id,
            messages=self.messages,
            toolConfig=self.tools,
            system=self.system,
            inferenceConfig=inference_config,
            additionalModelRequestFields=additional_model_request_fields
        )

        log.debug("Final request body: %s", json.dumps(
                request.model_dump(
                    exclude_none=True,
                    exclude={"response_prefix"}
                ),
                indent=4
            )
        )
        return request.model_dump_json(
            exclude_none=True, exclude={"response_prefix"}
        )


class BedrockConverseLLM(LLM):
    """An LLM using the Bedrock 'converse' operation."""
    client: BedrockRuntimeClient

    def __init__(
        self,
        model_id: str = CONVERSE_BEDROCK_MODEL_IDS["Claude 3 Haiku"],
        client: BedrockRuntimeClient | None = None,
    ) -> None:
        """Initialize.

        Args:
            model_id: Model ID. Defaults to the model ID for Claude 3 Haiku.
            client: Optional pre-initialized boto3 client. Defaults to None.
        """
        self.model_id = model_id
        self.client = client or boto3.client("bedrock-runtime") # type: ignore[reportUnknownMemberType]

    @timed_function
    def prompt(
        self,
        messages: Sequence[LLMMessage],
        inference_cfg: LLMInferenceConfig,
        *,
        auto_use_tools: bool = False,
    ) -> list[ConverseMessage]:
        """Prompt the LLM with a message and optional conversation history."""
        if not messages:
            msg = "Must specify at least one message."
            raise ValueError(msg)
        messages = [ConverseMessage.from_llm_message(m) for m in messages]
        request = ConverseInvokeLLMRequest.from_inference_config(inference_cfg, self.model_id, messages)  # noqa: E501
        response = self.invoke_model_with_request(request)
        if response.stop_reason == "tool_use" and auto_use_tools:
            assert request.tools is not None  # noqa: S101
            log.info("Tool-use requested:")
            log.info(response.content)
            tool_result_msgs = self.use_tools(response.content, request.tools)
            log.info("Tool-use results:")
            log.info(tool_result_msgs)
            tool_result_msgs = self.use_tools(response.content, request.tools)
            new_messages = [
                *messages,
                response.to_message(),
                *tool_result_msgs
            ]
            return self.prompt(
                new_messages,
                inference_cfg
            )
        return [*messages, response.to_message()]

    @timed_function
    def invoke_model_with_request(
        self, request: ConverseInvokeLLMRequest
    ) -> ConverseResponse:
        """Invoke model with request and get a response back."""
        response_body = self._make_client_request(request)
        converse_response = self._parse_response(response_body, request)
        return converse_response

    def use_tools(
        self,
        content: Sequence[ConverseTextContent | ConverseToolUseContent],
        tools: ConverseTools,
    ) -> list[ConverseUserMessage]:
        """Invoke tool-use blocks and return user messages."""
        tools_dict = {t.toolSpec.name: t.toolSpec for t in tools.tools}
        out_messages: list[ConverseUserMessage] = []
        for block in content:
            if not isinstance(block, ConverseToolUseContent):
                continue
            tool = tools_dict[block.toolUse.name]
            out_messages.append(tool.use(block))
        return out_messages

    def _parse_response(
            self, response_body: str, request: ConverseInvokeLLMRequest
        ) -> ConverseResponse:
        """Parse raw JSON response into a ConverseResponse."""
        #Can make this part smoother - ideally just .model_validate_json
        raw_data = json.loads(response_body)
        msg_data = raw_data.get("output", {}).get("message", {})
        msg_data["stopReason"] = raw_data.get("stopReason")
        msg_data["usage"] = raw_data.get("usage")

        response = ConverseResponse.model_validate(msg_data)
        if request.response_prefix is not None:
            response = self._add_prefix_to_response(
                response, request.response_prefix
            )
        return response

    def _make_client_request(self, request: ConverseInvokeLLMRequest) -> str:
        """Make model invocation request and return raw JSON response."""
        request_body = request.to_request_body()
        try:
            params = json.loads(request_body)
            response = self.client.converse(**params)
        except botocore.exceptions.ClientError as e:
            log.error("Failed with %s", e)  # noqa: TRY400
            log.error("Request body: %s", request_body)  # noqa: TRY400
            raise
        return json.dumps(response)

    def _add_prefix_to_response(
        self, response: ConverseResponse, prefix: str
    ) -> ConverseResponse:
        """Prepend the prefix to the first text block in the response."""
        for content_block in response.content:
            if isinstance(content_block, ConverseTextContent):
                content_block.text = prefix + content_block.text
                break
        return response
