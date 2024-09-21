import json

import boto3
import botocore.exceptions
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient

from e84_geoai_common.llm.core import LLM, InvokeLLMRequest
from e84_geoai_common.util import timed_function


class BedrockClaudeLLM(LLM):
    """Implements the LLM class for Bedrock Claude."""

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
    def invoke_model_with_request(self, request: InvokeLLMRequest) -> str:
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
