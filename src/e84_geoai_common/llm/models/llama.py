import logging
from collections.abc import Sequence
from typing import Any, Literal  # , cast

import boto3
import botocore.exceptions
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from pydantic import BaseModel, ConfigDict, Field

from e84_geoai_common.llm.core.llm import (
    LLM,
    LLMInferenceConfig,
    LLMMessage,
)
from e84_geoai_common.util import timed_function

log = logging.getLogger(__name__)

# # https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html#model-ids-arns
LLAMA_BEDROCK_MODEL_IDS = {
    "Llama 3 70B Instruct": "meta.llama3-70b-instruct-v1:0",
    # "Llama 3.2 11B Vision Instruct": "meta.llama3-2-11b-instruct-v1:0",
    # "Llama 3 8B Instruct": "meta.llama3-8b-instruct-v1:0",
}


class LlamaTextContent(BaseModel):
    """Llama text context model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    type: Literal["text"] = "text"
    text: str


class LlamaToolUseContent(BaseModel):
    """Represents a tool use request from Llama."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    type: Literal["tool_use"] = "tool_use"
    id: str
    input: dict[str, Any]
    name: str


LlamaMessageContentType = LlamaTextContent


class LlamaPrompt(BaseModel):
    """Llama prompt base model."""

    model_config = ConfigDict(strict=True, extra="forbid")
    prompt: str


class LlamaInvokeLLMRequest(BaseModel):
    """Represents a request to invoke Llama and get a response back."""

    model_config = ConfigDict(strict=True, extra="forbid")
    max_gen_len: int = Field(default=512, description="Maximum number of output tokens")
    prompt: str
    temperature: float = Field(
        default=0,
        description="Temperature control for randomness. Closer to zero = more deterministic.",
    )
    top_p: float | None = Field(
        default=0.9,
        description="Use a lower value to ignore less probable options",
    )


#################################################################################
# Other response objects


class LlamaResponse(BaseModel):
    """Llama response model."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    generation: str
    prompt_token_count: int
    generation_token_count: int
    stop_reason: Literal["stop", "length"]


def _llm_message_to_llama_prompt(
    system_prompt: str | None, messages: Sequence[LLMMessage]
) -> LlamaPrompt:
    """Converts the generic LLM Message into a LlamaPrompt."""
    messages_str = ""
    for msg in messages:
        if not isinstance(msg.content, str):
            raise TypeError("Unsupported ContentType")
        messages_str += f"{msg.content}\n"

    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>"
        f"{messages_str}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )

    return LlamaPrompt(prompt=prompt)


def _config_to_response_prefix(config: LLMInferenceConfig) -> str | None:
    if config.response_prefix:
        return config.response_prefix
    return None


class BedrockLlamaLLM(LLM):
    """Implements the LLM class for Bedrock Llama."""

    client: BedrockRuntimeClient

    def __init__(
        self,
        model_id: str = LLAMA_BEDROCK_MODEL_IDS["Llama 3 70B Instruct"],
        client: BedrockRuntimeClient | None = None,
    ) -> None:
        """Initialize.

        Args:
            model_id: Model ID. Defaults to the model ID for Llama 3 70B Instruct.
            client: Optional pre-initialized boto3 client. Defaults to None.
        """
        self.model_id = model_id
        self.client = client or boto3.client("bedrock-runtime")  # type: ignore[reportUnknownMemberType]

    def _create_request(
        self, messages: Sequence[LLMMessage], config: LLMInferenceConfig
    ) -> LlamaInvokeLLMRequest:
        response_prefix = _config_to_response_prefix(config)
        if response_prefix:
            messages = [*messages, LLMMessage(role="assistant", content=response_prefix)]

        llama_prompt = _llm_message_to_llama_prompt(config.system_prompt, messages)

        return LlamaInvokeLLMRequest(
            prompt=llama_prompt.prompt,
            max_gen_len=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )

    @timed_function
    def prompt(
        self,
        messages: Sequence[LLMMessage],
        inference_cfg: LLMInferenceConfig,
    ) -> LLMMessage:
        """Prompt the LLM with a message and optional conversation history."""
        if len(messages) == 0:
            raise ValueError("Must specify at least one message.")
        request = self._create_request(messages, inference_cfg)
        response = self.invoke_model_with_request(request)

        llm_msg = self._response_to_llm_message(response, inference_cfg)
        return llm_msg

    @timed_function
    def invoke_model_with_request(self, request: LlamaInvokeLLMRequest) -> LlamaResponse:
        """Invoke model with request and get a response back."""
        try:
            response = self.client.invoke_model(
                modelId=self.model_id, body=request.model_dump_json(exclude_none=True)
            )
        except botocore.exceptions.ClientError:
            log.exception("Request body: %s", request.model_dump_json())
            raise
        response_body = response["body"].read().decode("UTF-8")
        llama_response = LlamaResponse.model_validate_json(response_body)
        log.info(
            "Prompt token count: %s | Generation token count: %s",
            llama_response.prompt_token_count,
            llama_response.generation_token_count,
        )
        return llama_response

    def _response_to_llm_message(
        self, response: LlamaResponse, inference_cfg: LLMInferenceConfig
    ) -> LLMMessage:
        text = response.generation
        response_prefix = _config_to_response_prefix(inference_cfg)
        if response_prefix:
            text = response_prefix + text

        return LLMMessage(role="assistant", content=text)
