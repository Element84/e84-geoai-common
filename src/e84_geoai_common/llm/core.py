"""TODO"""

from abc import ABC, abstractmethod
import json
from typing import Generic, Literal, TypeVar

import boto3
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from pydantic import BaseModel, ConfigDict


class LLMMessage(BaseModel):
    """TODO"""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    role: Literal["user", "assistant"] = "user"
    # FUTURE: This could be changed to allow for multiple items following the anthropic content style
    content: str


class InvokeLLMRequest(BaseModel):
    """TODO"""

    model_config = ConfigDict(strict=True, extra="forbid")

    system: str | None = None
    max_tokens: int = 1000
    temperature: float = 0
    top_p: float = 0
    top_k: int = 0
    json_mode: bool = False
    messages: list[LLMMessage] = []


class LLM(ABC):
    """TODO"""

    @abstractmethod
    def invoke_model(self, request: InvokeLLMRequest) -> str:
        """TODO"""
        ...


class BedrockClaudeLLM(LLM):
    """TODO"""

    client: BedrockRuntimeClient

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0",
        client: BedrockRuntimeClient | None = None,
    ) -> None:
        self.model_id = model_id
        self.client = client or boto3.client("bedrock-runtime")  # type: ignore

    def _llm_request_to_body(self, request: InvokeLLMRequest) -> str:
        messages = [
            {"role": msg.role, "content": msg.content} for msg in request.messages
        ]
        if request.json_mode:
            # Force Claude into JSON mode
            messages.append({"role": "assistant", "content": "{"})

        return json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": request.max_tokens,
                "system": request.system,
                "messages": messages,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
            }
        )

    def invoke_model(self, request: InvokeLLMRequest) -> str:
        resp = self.client.invoke_model(
            modelId=self.model_id, body=self._llm_request_to_body(request)
        )
        body = str(resp["body"].read(), "UTF-8")

        if request.json_mode:
            return "{" + body
        else:
            return body


Model = TypeVar("Model", bound=BaseModel)


class ExtractDataRequest(InvokeLLMRequest, Generic[Model]):
    """TODO"""

    model: Model


def extract_data_from_text(llm: LLM, request: ExtractDataRequest[Model]) -> Model:
    """TODO"""
    # TODO ideally we wouldn't have to set this here.
    request.json_mode = True
    resp = llm.invoke_model(request)
    return request.model.model_validate_json(resp)
