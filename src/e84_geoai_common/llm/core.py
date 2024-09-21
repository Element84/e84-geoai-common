"""TODO"""

from abc import ABC, abstractmethod
import json
from typing import Generic, Literal, TypeVar

import boto3
import botocore.exceptions
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from pydantic import BaseModel, ConfigDict

from e84_geoai_common.util import timed_function


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

    @timed_function
    def invoke_model(self, request: InvokeLLMRequest) -> str:
        if len(request.messages) == 0:
            raise Exception("Must specify at least one message")
        req_body = self._llm_request_to_body(request)
        try:
            resp = self.client.invoke_model(modelId=self.model_id, body=req_body)
        except botocore.exceptions.ClientError as vex:
            print("Failed with", vex)
            print("Request body:", req_body)
            raise vex
        body = str(resp["body"].read(), "UTF-8")
        parsed = json.loads(body)
        llm_response = parsed["content"][0]["text"]
        if request.json_mode:
            return "{" + llm_response
        else:
            return llm_response


Model = TypeVar("Model", bound=BaseModel)


class ExtractDataExample(BaseModel, Generic[Model]):
    """TODO"""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    name: str
    user_query: str
    structure: Model

    def to_str(self) -> str:
        query_json = f"```json\n{self.structure.model_dump_json(indent=2, exclude_none=True)}\n```"
        return f'Example: {self.name}\nUser Query: "{self.user_query}"\n\n{query_json}'


def extract_data_from_text(
    *,
    llm: LLM,
    model_type: type[Model],
    system_prompt: str,
    user_prompt: str,
) -> Model:
    """TODO"""
    request = InvokeLLMRequest(
        system=system_prompt, json_mode=True, messages=[LLMMessage(content=user_prompt)]
    )
    resp = llm.invoke_model(request)
    return model_type.model_validate_json(resp)
