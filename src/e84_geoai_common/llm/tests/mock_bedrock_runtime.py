import json
import os
from io import BytesIO
from typing import Any, Unpack

import boto3
from botocore.response import StreamingBody
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from mypy_boto3_bedrock_runtime.type_defs import (
    ConverseRequestTypeDef,
    ConverseResponseTypeDef,
    InvokeModelRequestTypeDef,
    InvokeModelResponseTypeDef,
)

# Configures an override for if we'll run all tests with a real bedrock client. If not set to "true"
# we'll use a mock client
USE_REAL_BEDROCK_RUNTIME_CLIENT = os.getenv("USE_REAL_BEDROCK_RUNTIME_CLIENT") == "true"


def _string_to_streaming_body(string_data: str) -> StreamingBody:
    bytes_data = string_data.encode("utf-8")
    stream = BytesIO(bytes_data)
    return StreamingBody(stream, len(bytes_data))


def claude_response_with_content(
    content: str | list[dict[str, Any]], overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Creates a mock claude response with the given content."""
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    out = {
        "id": "msg_bdrk_123fake",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-haiku-20240307",
        "content": content,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 123, "output_tokens": 321},
    }
    if overrides is not None:
        out.update(overrides)
    return out


def nova_response_with_content(
    content: str | list[dict[str, Any]], overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Creates a mock Nova response with the given text."""
    if isinstance(content, str):
        content = [{"text": content}]
    out = {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 123,
            "outputTokens": 123,
            "totalTokens": 123,
            "cacheReadInputTokenCount": 123,
            "cacheWriteInputTokenCount": 123,
        },
    }
    if overrides is not None:
        out.update(overrides)
    return out


def converse_response_with_content(
    content: str | list[dict[str, Any]], overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Creates a mock Converse response with the given text."""
    if isinstance(content, str):
        content = [{"text": content}]
    out = {
        "additionalModelResponseFields": None,
        "metrics": {"latencyMs": 10},
        "output": {
            "message": {"role": "assistant", "content": content},
        },
        "performanceConfig": None,
        "ResponseMetadata": {},
        "role": "assistant",
        "stopReason": "end_turn",
        "trace": None,
        "usage": {
            "inputTokens": 123,
            "outputTokens": 123,
            "totalTokens": 123,
        },
    }
    if overrides is not None:
        out.update(overrides)
    return out


def titanv2_response(
    embedding: list[float], overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Creates a mock claude response with the given content."""
    out = {
        "embedding": embedding,
        "inputTextTokenCount": 300,
        "embeddingsByType": {
            "float": embedding,
        },
    }
    if overrides is not None:
        out.update(overrides)
    return out


class _MockBedrockRuntimeClient(BedrockRuntimeClient):
    """Implements the bedrock runtime client to return a set of canned responses."""

    canned_responses: list[dict[str, Any]]

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        """Creates the client with the given set of responses."""
        self.canned_responses = responses

    def invoke_model(
        self, **_kwargs: Unpack[InvokeModelRequestTypeDef]
    ) -> InvokeModelResponseTypeDef:
        """Overrides the invoke_model method to return the next canned response."""
        next_resp = self.canned_responses.pop(0)
        return {
            "body": _string_to_streaming_body(json.dumps(next_resp)),
            "contentType": "application/json",
            "performanceConfigLatency": "standard",
            "ResponseMetadata": {
                "RequestId": "123",
                "HTTPStatusCode": 200,
                "HTTPHeaders": {},
                "RetryAttempts": 0,
            },
        }

    def converse(self, **_kwargs: Unpack[ConverseRequestTypeDef]) -> ConverseResponseTypeDef:
        """Overrides the invoke_model method to return the next canned response."""
        # Pop the next canned response (expected to match `message` structure)
        next_resp = self.canned_responses.pop(0)
        return ConverseResponseTypeDef(**next_resp)


def make_test_bedrock_runtime_client(
    responses: list[dict[str, Any]] | None = None, *, use_real_client: bool = False
) -> BedrockRuntimeClient:
    """Creates a BedrockRuntimeClient for testing.

    Depending on configuration it will either use the canned responses or a real client.

    Args:
        responses: A list of dictionaries containing mock responses to be returned by the client.
            Required when not using a real client.
        use_real_client: If True, creates a real Bedrock client instead of a mock client.
            This overrides the USE_REAL_BEDROCK_RUNTIME_CLIENT environment variable.

    Returns:
        BedrockRuntimeClient: Either a real Bedrock client or a mock client with canned responses.

    Raises:
        RuntimeError: If not using a real client and no responses are provided.
    """
    if USE_REAL_BEDROCK_RUNTIME_CLIENT or use_real_client:
        return boto3.client("bedrock-runtime")  # type: ignore[reportUnknownMemberType]
    if responses is not None:
        return _MockBedrockRuntimeClient(responses)
    raise RuntimeError("If not using a real client the responses must be provided")
