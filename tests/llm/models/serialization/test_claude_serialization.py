from unittest.mock import Mock

from e84_geoai_common.llm.core.llm import (
    CachePointContent,
    LLMInferenceConfig,
    LLMMessage,
    TextContent,
)
from e84_geoai_common.llm.models.claude import BedrockClaudeLLM


def test_serialize_string_content() -> None:
    # Setup
    llm = BedrockClaudeLLM(client=Mock())
    message = LLMMessage(content="Output the word hello backwards and only that.")
    config = LLMInferenceConfig()

    # Execute
    request = llm.create_request([message], config)

    # Validate
    request_dict = request.model_dump(mode="json")
    assert request_dict["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Output the word hello backwards and only that.",
                    "cache_control": None,
                }
            ],
        }
    ]
    assert request.messages


def test_serialize_text_content() -> None:
    # Setup
    llm = BedrockClaudeLLM(client=Mock())
    message = LLMMessage(
        content=[TextContent(text="Output the word hello backwards and only that.")]
    )
    config = LLMInferenceConfig()

    # Execute
    request = llm.create_request([message], config)

    # Validate
    request_dict = request.model_dump(mode="json")
    assert request_dict["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Output the word hello backwards and only that.",
                    "cache_control": None,
                }
            ],
        }
    ]
    assert request.messages


def test_serialize_text_content_with_caching() -> None:
    # Setup
    llm = BedrockClaudeLLM(client=Mock())
    message = LLMMessage(
        content=[
            TextContent(text="Output the word hello backwards and only that."),
            CachePointContent(),
        ]
    )
    config = LLMInferenceConfig()

    # Execute
    request = llm.create_request([message], config)

    # Validate
    request_dict = request.model_dump(mode="json")
    assert request_dict["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Output the word hello backwards and only that.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]
    assert request.messages
