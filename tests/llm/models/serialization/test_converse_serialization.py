from unittest.mock import Mock

from e84_geoai_common.llm.core.llm import LLMInferenceConfig, LLMMessage, TextContent
from e84_geoai_common.llm.models.converse.converse import BedrockConverseLLM


def test_serialize_string_content() -> None:
    # Setup
    llm = BedrockConverseLLM(client=Mock())
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
                    "text": "Output the word hello backwards and only that.",
                },
            ],
        }
    ]
    assert request.messages


def test_serialize_text_content() -> None:
    # Setup
    llm = BedrockConverseLLM(client=Mock())
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
                    "text": "Output the word hello backwards and only that.",
                }
            ],
        }
    ]
    assert request.messages


def test_serialize_text_content_with_caching() -> None:
    # Setup
    llm = BedrockConverseLLM(client=Mock())
    message = LLMMessage(
        content=[
            TextContent(text="Output the word hello backwards and only that.", should_cache=True)
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
                    "text": "Output the word hello backwards and only that.",
                },
                {"cachePoint": {"type": "default"}},
            ],
        }
    ]
    assert request.messages
