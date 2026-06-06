import base64
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel, Field
from rich import print as rich_print

from e84_geoai_common.llm.core.llm import (
    Base64ImageContent,
    CachePointContent,
    ExecutableLLMTool,
    JSONContent,
    LLMInferenceConfig,
    LLMTool,
    LLMToolResultContent,
    LLMToolUseContent,
    LLMUserMessage,
    TextContent,
)
from e84_geoai_common.llm.models.claude import (
    CLAUDE_4_5_HAIKU,
    BedrockClaudeLLM,
    ClaudeTextContent,
)
from e84_geoai_common.llm.models.claude.streaming import (
    ClaudeInputJsonDelta,
    ClaudeStreamContentBlockDelta,
    ClaudeStreamContentBlockStart,
    ClaudeStreamContentBlockStop,
    ClaudeStreamMessageDelta,
    ClaudeStreamMessageStart,
    ClaudeStreamMessageStop,
    ClaudeStreamTextBlock,
    ClaudeStreamToolUseBlock,
    ClaudeTextDelta,
)
from e84_geoai_common.llm.tests.mock_bedrock_runtime import (
    MockAsyncEventStream,
    _MockBedrockRuntimeClient,  # type: ignore[reportPrivateUsage]
    claude_response_with_content,
    claude_streaming_events_for_text,
    claude_streaming_events_for_tool_use,
    make_test_bedrock_runtime_client,
)
from tests.llm.models.utils import build_long_system_prompt


def test_basic_usage() -> None:
    llm = BedrockClaudeLLM(
        client=make_test_bedrock_runtime_client([claude_response_with_content("olleh")])
    )
    config = LLMInferenceConfig()
    resp = llm.prompt(
        [LLMUserMessage(content="Output the word hello backwards and only that.")], config
    )
    assert resp.model_dump(exclude={"metadata": {"input_tokens", "output_tokens"}}) == {
        "role": "assistant",
        "content": [{"text": "olleh"}],
        "metadata": {"stop_reason": "end_turn"},
    }


def encode_image_to_base64_str(image_path: str | Path) -> str:
    image_path = Path(image_path)
    with image_path.open("rb") as image_file:
        encoded_bytes = base64.b64encode(image_file.read())
    return encoded_bytes.decode("utf-8")


def test_image_input() -> None:
    llm = BedrockClaudeLLM(
        client=make_test_bedrock_runtime_client([claude_response_with_content("cat")])
    )

    # locally ai generated picture of a cat
    image_path = Path(__file__).parent / "images/cat.webp"
    base64_string = encode_image_to_base64_str(image_path)

    image_content = Base64ImageContent(media_type="image/webp", data=base64_string)
    prompt_text = TextContent(
        text="Report the animal in the picture and only that, in lowercase. I.e. dog"
    )

    prompt_message = LLMUserMessage(content=[prompt_text, image_content])

    config = LLMInferenceConfig()

    resp = llm.prompt([prompt_message], config)

    assert resp.model_dump(exclude={"metadata": {"input_tokens", "output_tokens"}}) == {
        "role": "assistant",
        "content": [{"text": "cat"}],
        "metadata": {"stop_reason": "end_turn"},
    }


def test_tool_use_json() -> None:
    prompt = "What's the weather like in Philadelphia right now?"

    # tool input
    class GetWeatherInfoInput(BaseModel):
        place: str = Field(description="Place to look up weather for.")

    # tool output
    class WeatherInfo(BaseModel):
        """Weather info."""

        weather_description: str = Field(description="Sunny, cloudy, raining etc.")
        temperature: float = Field(description="Temperature in Celsius")
        humidity: float = Field(description="Humidity %.")

    class _WeatherTool(ExecutableLLMTool[None]):
        def _execute(
            self,
            context: None,  # noqa: ARG002
            tool_use_request: LLMToolUseContent,
        ) -> LLMToolResultContent:
            weather_info = WeatherInfo(weather_description="Sunny", temperature=20, humidity=50)
            tool_result = LLMToolResultContent(
                id=tool_use_request.id, content=[JSONContent(data=weather_info.model_dump())]
            )
            return tool_result

    tool = _WeatherTool(
        tool_spec=LLMTool(
            name="GetWeatherInfo",
            description="Get current weather info for a place.",
            input_model=GetWeatherInfoInput,
            output_model=WeatherInfo,
        )
    )

    dummy_responses = [
        # response to initial question
        claude_response_with_content(
            [
                {"text": "To answer this question I will use the GetWeatherInfo tool."},
                {
                    "id": "abc123",
                    "name": "GetWeatherInfo",
                    "input": {"place": "Philadelphia"},
                },
            ],
            overrides={"stop_reason": "tool_use"},
        ),
        # response to tool result
        claude_response_with_content(
            [
                {"text": "It's always sunny in Philadelphia."},
            ],
        ),
    ]
    client = make_test_bedrock_runtime_client(dummy_responses)
    llm = BedrockClaudeLLM(client=client)
    config = LLMInferenceConfig(tools=[tool.tool_spec])

    # test tool use
    messages = [LLMUserMessage(content=prompt)]
    resp = llm.prompt(messages=messages, inference_cfg=config)

    # print out solution if doing a live test, so we can inspect it if the test fails
    if not isinstance(client, _MockBedrockRuntimeClient):
        rich_print(resp)

    assert isinstance(resp.content, list)
    tool_use_req = resp.content[-1]
    assert isinstance(tool_use_req, LLMToolUseContent)
    assert tool_use_req.name == tool.tool_spec.name
    tool_inputs = GetWeatherInfoInput.model_validate(tool_use_req.input)
    assert "philadelphia" in tool_inputs.place.lower()

    # test tool result
    tool_result = tool.execute(None, tool_use_req)
    messages = [*messages, resp, LLMUserMessage(content=[tool_result])]
    resp = llm.prompt(messages=messages, inference_cfg=config)

    # print out solution if doing a live test, so we can inspect it if the test fails
    if not isinstance(client, _MockBedrockRuntimeClient):
        rich_print(resp)

    assert "sunny" in resp.to_text_only().lower()


def test_tool_use_image() -> None:
    prompt = "Generate a picture of a cat and briefly describe it."

    # tool input
    class ImageGeneratorInput(BaseModel):
        description: str = Field(description="Description of the image to generate.")

    class _ImageGeneratorTool(ExecutableLLMTool[None]):
        def _execute(
            self,
            context: None,  # noqa: ARG002
            tool_use_request: LLMToolUseContent,
        ) -> LLMToolResultContent:
            image_path = str(Path(__file__).parent / "images/cat.webp")
            base64_string = encode_image_to_base64_str(image_path)
            image_content = Base64ImageContent(media_type="image/webp", data=base64_string)
            tool_result = LLMToolResultContent(id=tool_use_request.id, content=[image_content])
            return tool_result

    tool = _ImageGeneratorTool(
        tool_spec=LLMTool(
            name="GenerateImage",
            description="Generate an image from text. Returns the generated image only.",
            input_model=ImageGeneratorInput,
            output_model=None,
        ),
    )

    dummy_responses = [
        # response to initial question
        claude_response_with_content(
            [
                {"text": "Generating an image using the tool."},
                {
                    "id": "abc123",
                    "name": "GenerateImage",
                    "input": {"description": "A cat"},
                },
            ],
            overrides={"stop_reason": "tool_use"},
        ),
        # response to tool result
        claude_response_with_content(
            [
                {"text": "The image contains a cat."},
            ],
        ),
    ]
    client = make_test_bedrock_runtime_client(dummy_responses)
    llm = BedrockClaudeLLM(client=client)
    config = LLMInferenceConfig(tools=[tool.tool_spec])

    # test tool use
    messages = [LLMUserMessage(content=prompt)]
    resp = llm.prompt(messages=messages, inference_cfg=config)

    # print out solution if doing a live test, so we can inspect it if the test fails
    if not isinstance(client, _MockBedrockRuntimeClient):
        rich_print(resp)

    assert isinstance(resp.content, list)
    tool_use_req = resp.content[-1]
    assert isinstance(tool_use_req, LLMToolUseContent)
    assert tool_use_req.name == tool.tool_spec.name
    _ = ImageGeneratorInput.model_validate(tool_use_req.input)

    # test tool result
    tool_result = tool.execute(None, tool_use_req)
    messages = [*messages, resp, LLMUserMessage(content=[tool_result])]
    resp = llm.prompt(messages=messages, inference_cfg=config)

    # print out solution if doing a live test, so we can inspect it if the test fails
    if not isinstance(client, _MockBedrockRuntimeClient):
        rich_print(resp)

    assert "cat" in resp.to_text_only().lower()


def test_basic_usage_with_prompt_caching() -> None:
    text_content = TextContent(text="Output the word hello backwards and only that.")
    llm = BedrockClaudeLLM(
        client=make_test_bedrock_runtime_client(
            [claude_response_with_content([{"type": "text", "text": "olleh"}])]
        )
    )
    config = LLMInferenceConfig()
    resp = llm.prompt([LLMUserMessage(content=[text_content, CachePointContent()])], config)
    assert resp.model_dump(exclude={"metadata": {"input_tokens", "output_tokens"}}) == {
        "role": "assistant",
        "content": [{"text": "olleh"}],
        "metadata": {"stop_reason": "end_turn"},
    }


def test_large_system_prompt() -> None:
    """This test is most interesting as a live-test.

    It validates that the cache control headers indicate that prompt caching actually worked.

    It uses a large system prompt so that the minimum token limit is reached.
    """
    long_system_prompt = build_long_system_prompt()

    text_content = TextContent(text="Output the word hello backwards and only that.")
    llm = BedrockClaudeLLM(
        model_id=CLAUDE_4_5_HAIKU,
        client=make_test_bedrock_runtime_client(
            [
                claude_response_with_content(
                    [{"type": "text", "text": "olleh"}],
                    {
                        "usage": {
                            "input_tokens": 123,
                            "output_tokens": 321,
                            "cache_creation_input_tokens": 2500,
                            "cache_read_input_tokens": 0,
                        }
                    },
                )
            ]
        ),
    )
    config = LLMInferenceConfig(system_prompt=long_system_prompt)
    request = llm.create_request(
        [LLMUserMessage(content=[text_content, CachePointContent()])], config
    )
    response = llm.invoke_model_with_request(request)

    assert response.content == [ClaudeTextContent(text="olleh")]
    assert response.usage.cache_creation_input_tokens is not None
    assert response.usage.cache_creation_input_tokens > 0


# =============================================================================
# Streaming tests
# =============================================================================


@pytest.mark.asyncio
async def test_prompt_stream_text() -> None:
    """Test that prompt_stream yields the correct typed events for a text response."""
    events = claude_streaming_events_for_text("Hello, world!")
    mock_event_stream = MockAsyncEventStream(events)

    mock_response = {"body": mock_event_stream}
    mock_client_cm = AsyncMock()
    mock_client_cm.__aenter__ = AsyncMock(return_value=mock_client_cm)
    mock_client_cm.__aexit__ = AsyncMock(return_value=False)
    mock_client_cm.invoke_model_with_response_stream = AsyncMock(return_value=mock_response)

    llm = BedrockClaudeLLM(
        client=make_test_bedrock_runtime_client([claude_response_with_content("unused")]),
        region_name="us-east-1",
    )

    with patch("e84_geoai_common.llm.models.claude.claude.aioboto3") as mock_aioboto3:
        mock_session = mock_aioboto3.Session.return_value
        mock_session.client.return_value = mock_client_cm

        collected = [
            event
            async for event in llm.prompt_stream(
                [LLMUserMessage(content="Hello")], LLMInferenceConfig()
            )
        ]

    # Verify event sequence (ping should be skipped)
    assert len(collected) == 6
    assert isinstance(collected[0], ClaudeStreamMessageStart)
    assert collected[0].message.id == "msg_stream_123"
    assert collected[0].message.model == "claude-3-haiku-20240307"

    assert isinstance(collected[1], ClaudeStreamContentBlockStart)
    assert collected[1].index == 0
    assert isinstance(collected[1].content_block, ClaudeStreamTextBlock)

    assert isinstance(collected[2], ClaudeStreamContentBlockDelta)
    assert collected[2].index == 0
    assert isinstance(collected[2].delta, ClaudeTextDelta)
    assert collected[2].delta.text == "Hello, world!"

    assert isinstance(collected[3], ClaudeStreamContentBlockStop)
    assert collected[3].index == 0

    assert isinstance(collected[4], ClaudeStreamMessageDelta)
    assert collected[4].delta.stop_reason == "end_turn"
    assert collected[4].usage.output_tokens == 15

    assert isinstance(collected[5], ClaudeStreamMessageStop)


@pytest.mark.asyncio
async def test_prompt_stream_tool_use() -> None:
    """Test that prompt_stream yields correct typed events for tool use."""
    events = claude_streaming_events_for_tool_use(
        tool_name="get_weather",
        tool_id="toolu_abc123",
        input_json='{"location": "San Francisco, CA"}',
    )
    mock_event_stream = MockAsyncEventStream(events)

    mock_response = {"body": mock_event_stream}
    mock_client_cm = AsyncMock()
    mock_client_cm.__aenter__ = AsyncMock(return_value=mock_client_cm)
    mock_client_cm.__aexit__ = AsyncMock(return_value=False)
    mock_client_cm.invoke_model_with_response_stream = AsyncMock(return_value=mock_response)

    llm = BedrockClaudeLLM(
        client=make_test_bedrock_runtime_client([claude_response_with_content("unused")]),
        region_name="us-east-1",
    )

    with patch("e84_geoai_common.llm.models.claude.claude.aioboto3") as mock_aioboto3:
        mock_session = mock_aioboto3.Session.return_value
        mock_session.client.return_value = mock_client_cm

        collected = [
            event
            async for event in llm.prompt_stream(
                [LLMUserMessage(content="What's the weather?")],
                LLMInferenceConfig(
                    tools=[
                        LLMTool(
                            name="get_weather",
                            description="Get weather",
                            input_model=None,
                            output_model=None,
                        )
                    ]
                ),
            )
        ]

    # Verify event sequence (2 input_json_delta events due to chunking)
    assert len(collected) == 7
    assert isinstance(collected[0], ClaudeStreamMessageStart)

    assert isinstance(collected[1], ClaudeStreamContentBlockStart)
    assert isinstance(collected[1].content_block, ClaudeStreamToolUseBlock)
    assert collected[1].content_block.name == "get_weather"
    assert collected[1].content_block.id == "toolu_abc123"

    # Two input_json_delta events
    assert isinstance(collected[2], ClaudeStreamContentBlockDelta)
    assert isinstance(collected[2].delta, ClaudeInputJsonDelta)
    assert isinstance(collected[3], ClaudeStreamContentBlockDelta)
    assert isinstance(collected[3].delta, ClaudeInputJsonDelta)

    assert isinstance(collected[4], ClaudeStreamContentBlockStop)

    assert isinstance(collected[5], ClaudeStreamMessageDelta)
    assert collected[5].delta.stop_reason == "tool_use"

    assert isinstance(collected[6], ClaudeStreamMessageStop)


@pytest.mark.asyncio
async def test_prompt_stream_skips_ping() -> None:
    """Test that ping events are silently skipped."""
    # Only ping + message_stop
    events = [
        {"type": "ping"},
        {"type": "ping"},
        {
            "type": "message_start",
            "message": {
                "id": "msg_ping_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-haiku-20240307",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        },
        {"type": "message_stop"},
    ]
    mock_event_stream = MockAsyncEventStream(events)

    mock_response = {"body": mock_event_stream}
    mock_client_cm = AsyncMock()
    mock_client_cm.__aenter__ = AsyncMock(return_value=mock_client_cm)
    mock_client_cm.__aexit__ = AsyncMock(return_value=False)
    mock_client_cm.invoke_model_with_response_stream = AsyncMock(return_value=mock_response)

    llm = BedrockClaudeLLM(
        client=make_test_bedrock_runtime_client([claude_response_with_content("unused")]),
        region_name="us-east-1",
    )

    with patch("e84_geoai_common.llm.models.claude.claude.aioboto3") as mock_aioboto3:
        mock_session = mock_aioboto3.Session.return_value
        mock_session.client.return_value = mock_client_cm

        collected = [
            event
            async for event in llm.prompt_stream(
                [LLMUserMessage(content="Hi")], LLMInferenceConfig()
            )
        ]

    # Pings should be skipped, only message_start and message_stop remain
    assert len(collected) == 2
    assert isinstance(collected[0], ClaudeStreamMessageStart)
    assert isinstance(collected[1], ClaudeStreamMessageStop)
