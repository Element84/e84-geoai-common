import json
from io import BytesIO
from typing import Any, Unpack

from botocore.response import StreamingBody
from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
from mypy_boto3_bedrock_runtime.type_defs import (
    InvokeModelRequestRequestTypeDef,
    InvokeModelResponseTypeDef,
)


def _string_to_streaming_body(string_data: str) -> StreamingBody:
    bytes_data = string_data.encode("utf-8")
    stream = BytesIO(bytes_data)
    return StreamingBody(stream, len(bytes_data))


def claude_response_with_content(text: str) -> dict[str, Any]:
    """Creates a mock claude response with the given text."""
    return {
        "id": "msg_bdrk_123fake",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-haiku-20240307",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 123, "output_tokens": 321},
    }


class MockBedrockRuntimeClient(BedrockRuntimeClient):
    """Implements the bedrock runtime client to return a set of canned responses."""

    canned_responses: list[dict[str, Any]]

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        """Creates the client with the given set of responses."""
        self.canned_responses = responses

    def invoke_model(
        self, **_kwargs: Unpack[InvokeModelRequestRequestTypeDef]
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
