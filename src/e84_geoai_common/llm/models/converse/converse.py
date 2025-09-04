import json
import logging
from collections.abc import Sequence
from typing import Any, cast

import boto3
import botocore.exceptions
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from mypy_boto3_bedrock_runtime.type_defs import ConverseResponseTypeDef
from pydantic import BaseModel, ConfigDict, Field

from e84_geoai_common.llm.core.llm import (
    LLM,
    Base64ImageContent,
    CachePointContent,
    JSONContent,
    LLMDataContentType,
    LLMInferenceConfig,
    LLMMessage,
    LLMMessageContentType,
    LLMMessageMetadata,
    LLMTool,
    LLMToolChoice,
    LLMToolResultContent,
    LLMToolUseContent,
    TextContent,
)
from e84_geoai_common.llm.models.claude import (
    CLAUDE_3_5_HAIKU,
    CLAUDE_3_5_SONNET,
    CLAUDE_3_5_SONNET_V2,
    CLAUDE_3_HAIKU,
    CLAUDE_3_OPUS,
    CLAUDE_3_SONNET,
    CLAUDE_4_SONNET,
    CLAUDE_INSTANT,
)
from e84_geoai_common.llm.models.converse.data_content_types import (
    ConverseCachePoint,
    ConverseImageContent,
    ConverseJSONContent,
    ConverseTextContent,
)
from e84_geoai_common.llm.models.converse.msg_types import (
    ConverseMessage,
    ConverseMessageContentType,
)
from e84_geoai_common.llm.models.converse.response_types import (
    ConverseAdditionalModelRequestFields,
    ConverseResponse,
    SystemContentBlock,
)
from e84_geoai_common.llm.models.converse.tool_use_types import (
    ConverseAnyToolChoice,
    ConverseAutoToolChoice,
    ConverseSingleTool,
    ConverseSpecificTool,
    ConverseSpecificToolChoice,
    ConverseToolChoiceType,
    ConverseToolConfig,
    ConverseToolInputSchema,
    ConverseToolResult,
    ConverseToolResultContent,
    ConverseToolSpec,
    ConverseToolUse,
    ConverseToolUseContent,
)
from e84_geoai_common.llm.models.nova import NOVA_LITE, NOVA_MICRO, NOVA_PRO
from e84_geoai_common.util import timed_function

# Converse uses camel case for its variables. Ignore any linting problems with this.
# ruff: noqa: N815

log = logging.getLogger(__name__)


LLAMA_3_1_70_B_INSTRUCT = "us.meta.llama3-1-70b-instruct-v1:0"
LLAMA_3_1_8_B_INSTRUCT = "us.meta.llama3-1-8b-instruct-v1:0"
LLAMA_3_2_11_B_VISION_INSTRUCT = "us.meta.llama3-2-11b-instruct-v1:0"
LLAMA_3_2_1_B_INSTRUCT = "us.meta.llama3-2-1b-instruct-v1:0"
LLAMA_3_2_3_B_INSTRUCT = "us.meta.llama3-2-3b-instruct-v1:0"
LLAMA_3_2_90_B_VISION_INSTRUCT = "us.meta.llama3-2-90b-instruct-v1:0"
LLAMA_3_3_70_B_INSTRUCT = "us.meta.llama3-3-70b-instruct-v1:0"

# DEPRECATED: Use the constants above instead.
CONVERSE_BEDROCK_MODEL_IDS = {
    "Claude 3 Haiku": CLAUDE_3_HAIKU,
    "Claude 3.5 Sonnet": CLAUDE_3_5_SONNET,
    "Claude 3 Sonnet": CLAUDE_3_SONNET,
    "Claude 3 Opus": CLAUDE_3_OPUS,
    "Claude Instant": CLAUDE_INSTANT,
    "Claude 3.5 Haiku": CLAUDE_3_5_HAIKU,
    "Claude 3.5 Sonnet v2": CLAUDE_3_5_SONNET_V2,
    "Claude 4 Sonnet": CLAUDE_4_SONNET,
    "Nova Lite": NOVA_LITE,
    "Nova Micro": NOVA_MICRO,
    "Nova Pro": NOVA_PRO,
    "Llama 3.1 70B Instruct": LLAMA_3_1_70_B_INSTRUCT,
    "Llama 3.1 8B Instruct": LLAMA_3_1_8_B_INSTRUCT,
    "Llama 3.2 11B Vision Instruct": LLAMA_3_2_11_B_VISION_INSTRUCT,
    "Llama 3.2 1B Instruct": LLAMA_3_2_1_B_INSTRUCT,
    "Llama 3.2 3B Instruct": LLAMA_3_2_3_B_INSTRUCT,
    "Llama 3.2 90B Vision Instruct": LLAMA_3_2_90_B_VISION_INSTRUCT,
    "Llama 3.3 70B Instruct": LLAMA_3_3_70_B_INSTRUCT,
}


class ConverseInferenceConfig(BaseModel):
    """Converse inference config model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    maxTokens: int | None
    stopSequences: Sequence[str] | None
    temperature: float | None
    topP: float | None


class ConverseInvokeLLMRequest(BaseModel):
    """Converse request model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    additionalModelRequestFields: ConverseAdditionalModelRequestFields | None = Field(
        default=None, description="Configuration for any additionoal fields models may need"
    )

    inferenceConfig: ConverseInferenceConfig | None = Field(
        default=None, description="Configuration of maxTokens, stopSequences, temperature, topP"
    )

    modelId: str = Field(
        default=CONVERSE_BEDROCK_MODEL_IDS["Claude 3 Haiku"],
        description="Model used for the Converse api",
    )

    messages: list[ConverseMessage] = Field(
        default_factory=list[ConverseMessage], description="List of LLM Messages"
    )

    system: Sequence[SystemContentBlock] | None = Field(default=None, description="System Prompt")

    toolConfig: ConverseToolConfig | None = Field(
        default=None, description="List of tools that the model may call."
    )


def _llm_message_to_converse_message(msg: LLMMessage) -> ConverseMessage:
    """Converts the generic LLM Message into a ConverseMessage."""

    def _handle_content(content: LLMMessageContentType) -> ConverseMessageContentType:
        """Convert `LLMMessageContentType` to a list of `ConverseMessageContentType`.

        This function returns a list the AWS Bedrock Converse API treats a
        cache point as its own content type, whereas this library's API treats
        it as a property of a content type.

        If this function receives a content instance whose property indicates
        it should request a cache point, it will return 2 content instances:
        one with the content of the message, and another with the cache point.
        """
        match content:
            case TextContent():
                return ConverseTextContent(text=content.text)
            case Base64ImageContent():
                return ConverseImageContent.from_b64_image_content(content)
            case LLMToolUseContent():
                return ConverseToolUseContent(
                    toolUse=ConverseToolUse(
                        toolUseId=content.id, name=content.name, input=content.input
                    )
                )

            case LLMToolResultContent():
                return _llm_tool_result_to_converse_tool_result(content)
            case CachePointContent():
                return ConverseCachePoint()

    if isinstance(msg.content, str):
        content = [ConverseTextContent(text=msg.content)]
    else:
        content = [_handle_content(content) for content in msg.content]
    return ConverseMessage(role=msg.role, content=content)


def _llm_tool_to_converse_tool(tool: LLMTool) -> ConverseSingleTool:
    """Build a ConverseSingleTool from an LLMTool.

    If LLMTool.output_model is set, the JSON schema of the output model is
    injected into the tool's description so that the LLM is aware of it.
    """
    if tool.input_model is None:
        input_schema = cast("dict[str, Any]", {"type": "object", "properties": {}})
    else:
        input_schema = tool.input_model.model_json_schema()

    description = tool.description
    if tool.output_model is not None:
        output_schema = json.dumps(tool.output_model.model_json_schema(), indent=2)
        description = f"{tool.description}\n\nOutput schema:\n```json\n{output_schema}\n```"

    converse_tool = ConverseSingleTool(
        toolSpec=ConverseToolSpec(
            name=tool.name,
            description=description,
            inputSchema=ConverseToolInputSchema(json=input_schema),
        )
    )
    return converse_tool


def _llm_tool_choice_to_converse_tool_choice(
    tool_choice: LLMToolChoice | None,
) -> ConverseToolChoiceType:
    if tool_choice is None:
        return ConverseAutoToolChoice()
    match tool_choice.mode:
        case "optional":
            return ConverseAutoToolChoice()
        case "force_tool_use":
            return ConverseAnyToolChoice()
        case "force_specific_tool_use":
            if tool_choice.tool_name is None:
                raise ValueError("Tool name not specified.")
            return ConverseSpecificToolChoice(tool=ConverseSpecificTool(name=tool_choice.tool_name))


def _llm_tool_result_to_converse_tool_result(
    tool_result: LLMToolResultContent,
) -> ConverseToolResultContent:
    def _to_tool_result_content(
        in_content: LLMDataContentType,
    ) -> ConverseTextContent | ConverseJSONContent | ConverseImageContent:
        match in_content:
            case TextContent():
                out_content = ConverseTextContent(text=in_content.text)
            case JSONContent():
                out_content = ConverseJSONContent(json=in_content.data)
            case Base64ImageContent():
                out_content = ConverseImageContent.from_b64_image_content(in_content)
        return out_content

    out_content = [_to_tool_result_content(c) for c in tool_result.content]
    out = ConverseToolResultContent(
        toolResult=ConverseToolResult(
            toolUseId=tool_result.id,
            status=tool_result.status,
            content=out_content,
        )
    )
    return out


class BedrockConverseLLM(LLM):
    """Implements the LLM class for Bedrock Converse."""

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
        self.client = client or boto3.client("bedrock-runtime")  # type: ignore[reportUnknownMemberType]

    def create_request(
        self, messages: Sequence[LLMMessage], config: LLMInferenceConfig
    ) -> ConverseInvokeLLMRequest:
        stop_sequences = None
        if config.json_mode:
            # https://docs.aws.amazon.com/nova/latest/userguide/prompting-structured-output.html
            prefix = "```json\n{"
            messages = [*messages, LLMMessage(role="assistant", content=prefix)]
            stop_sequences = ["```"]
        elif config.response_prefix:
            messages = [*messages, LLMMessage(role="assistant", content=config.response_prefix)]

        system = None
        if config.system_prompt:
            # For now, just support for a single system prompt content block
            system = [SystemContentBlock(text=config.system_prompt)]

        inference_config = ConverseInferenceConfig(
            maxTokens=config.max_tokens,
            stopSequences=stop_sequences,
            temperature=config.temperature,
            topP=config.top_p,
        )

        additional_model_request_fields = None
        if config.top_k:
            additional_model_request_fields = ConverseAdditionalModelRequestFields(
                top_k=config.top_k
            )

        tool_config = None
        if config.tools is not None:
            tools = [_llm_tool_to_converse_tool(t) for t in config.tools]
            tool_choice = _llm_tool_choice_to_converse_tool_choice(config.tool_choice)
            tool_config = ConverseToolConfig(tools=tools, toolChoice=tool_choice)

        request = ConverseInvokeLLMRequest(
            modelId=self.model_id,
            messages=[_llm_message_to_converse_message(msg) for msg in messages],
            toolConfig=tool_config,
            system=system,
            inferenceConfig=inference_config,
            additionalModelRequestFields=additional_model_request_fields,
        )

        return request

    @timed_function
    def prompt(
        self,
        messages: Sequence[LLMMessage],
        inference_cfg: LLMInferenceConfig,
    ) -> LLMMessage:
        """Prompt the LLM with a message and optional conversation history."""
        if not messages:
            msg = "Must specify at least one message."
            raise ValueError(msg)
        request = self.create_request(messages=messages, config=inference_cfg)
        response = self.invoke_model_with_request(request)
        llm_msg = self._response_to_llm_message(response, inference_cfg=inference_cfg)
        return llm_msg

    @timed_function
    def invoke_model_with_request(self, request: ConverseInvokeLLMRequest) -> ConverseResponse:
        """Invoke model with request and get a response back."""
        response_body = self._make_client_request(request)
        response = ConverseResponse.model_validate(response_body)
        log.info("Token usage: %s", response.usage)
        return response

    def _response_to_llm_message(
        self, response: ConverseResponse, inference_cfg: LLMInferenceConfig
    ) -> LLMMessage:
        def _to_llm_content(
            index: int,
            c: ConverseTextContent | ConverseImageContent | ConverseToolUseContent,
        ) -> TextContent | Base64ImageContent | LLMToolUseContent:
            match c:
                case ConverseTextContent():
                    content = c.text
                    if index == 0 and inference_cfg.response_prefix:
                        content = inference_cfg.response_prefix + content
                    return TextContent(text=content)
                case ConverseImageContent():
                    return c.to_b64_image_content()
                case ConverseToolUseContent():
                    return LLMToolUseContent(
                        id=c.toolUse.toolUseId,
                        name=c.toolUse.name,
                        input=c.toolUse.input,
                    )

        response_msg = response.output.message

        if len(response_msg.content) == 1 and isinstance(
            response_msg.content[0], ConverseTextContent
        ):
            content = response_msg.content[0].text
            if inference_cfg.json_mode:
                content = [TextContent(text="{" + content.removesuffix("```"))]
            elif inference_cfg.response_prefix:
                content = [TextContent(text=inference_cfg.response_prefix + content)]
            else:
                content = [TextContent(text=content)]
        else:
            content = [_to_llm_content(index, c) for index, c in enumerate(response_msg.content)]
        return LLMMessage(
            role=response_msg.role,
            content=content,
            metadata=LLMMessageMetadata(
                input_tokens=response.usage.inputTokens,
                output_tokens=response.usage.outputTokens,
                stop_reason=response.stopReason,
            ),
        )

    def _make_client_request(self, request: ConverseInvokeLLMRequest) -> ConverseResponseTypeDef:
        """Make model invocation request and return raw JSON response."""
        try:
            params = request.model_dump(exclude_none=True)
            response = self.client.converse(**params)
        except botocore.exceptions.ClientError as e:
            log.error("Failed with %s", e)  # noqa: TRY400
            log.error("Request body: %s", request)  # noqa: TRY400
            raise
        return response

    def _add_prefix_to_response(self, response: ConverseResponse, prefix: str) -> ConverseResponse:
        """Prepend the prefix to the first text block in the response."""
        for content_block in response.output.message.content:
            if isinstance(content_block, ConverseTextContent):
                content_block.text = prefix + content_block.text
                break
        return response
