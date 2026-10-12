import base64
from pathlib import Path

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
    LLMToolChoice,
    LLMToolResultContent,
    LLMToolUseContent,
    LLMUserMessage,
    TextContent,
)
from e84_geoai_common.llm.models.claude import (
    CLAUDE_4_5_HAIKU,
    CLAUDE_5_5_HAIKU,
    CLAUDE_5_5_OPUS,
    BedrockClaudeLLM,
    ClaudeTextContent,
    ClaudeToolChoice,
)
from e84_geoai_common.llm.tests.mock_bedrock_runtime import (
    _MockBedrockRuntimeClient,  # type: ignore[reportPrivateUsage]
    claude_response_with_content,
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


def test_thinking_blocks_are_dropped() -> None:
    llm = BedrockClaudeLLM(
        model_id=CLAUDE_5_5_HAIKU,
        client=make_test_bedrock_runtime_client(
            [
                claude_response_with_content(
                    [
                        {"type": "thinking", "thinking": "", "signature": "abc123"},
                        {"type": "redacted_thinking", "data": "encrypted"},
                        {"type": "text", "text": "olleh"},
                    ]
                )
            ]
        ),
    )
    resp = llm.prompt(
        [LLMUserMessage(content="Output the word hello backwards and only that.")],
        LLMInferenceConfig(max_tokens=4000),
    )
    assert resp.model_dump(exclude={"metadata": {"input_tokens", "output_tokens"}}) == {
        "role": "assistant",
        "content": [{"text": "olleh"}],
        "metadata": {"stop_reason": "end_turn"},
    }


def test_sampling_params_omitted_for_newer_models() -> None:
    messages = [LLMUserMessage(content="hello")]
    old_llm = BedrockClaudeLLM(
        model_id=CLAUDE_4_5_HAIKU, client=make_test_bedrock_runtime_client([])
    )
    assert old_llm.create_request(messages, LLMInferenceConfig()).temperature == 0

    for model_id in [CLAUDE_5_5_OPUS, "global.anthropic.claude-opus-5-5"]:
        llm = BedrockClaudeLLM(model_id=model_id, client=make_test_bedrock_runtime_client([]))
        assert llm.create_request(messages, LLMInferenceConfig()).temperature is None
        with pytest.raises(ValueError, match="not supported"):
            llm.create_request(messages, LLMInferenceConfig(temperature=0.5))
        with pytest.raises(ValueError, match="not supported"):
            llm.create_request(messages, LLMInferenceConfig(top_p=0.5))


class Country(BaseModel):
    name: str


class City(BaseModel):
    name: str
    country: Country


def test_structured_output_schema_disallows_additional_properties() -> None:
    llm = BedrockClaudeLLM(client=make_test_bedrock_runtime_client([]))
    request = llm.create_request(
        [LLMUserMessage(content="hello")], LLMInferenceConfig(structured_output_model=City)
    )
    assert request.output_config is not None
    schema = request.output_config.format.schema_
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["Country"]["additionalProperties"] is False


def test_forced_tool_choice_replaced_for_models_that_reject_it() -> None:
    messages = [LLMUserMessage(content="hello")]
    config = LLMInferenceConfig(
        system_prompt="Be helpful.",
        tools=[
            LLMTool(
                name="parse_data",
                description="Parse the data.",
                input_model=City,
                output_model=None,
            )
        ],
        tool_choice=LLMToolChoice(mode="force_specific_tool_use", tool_name="parse_data"),
    )

    old_llm = BedrockClaudeLLM(
        model_id=CLAUDE_4_5_HAIKU, client=make_test_bedrock_runtime_client([])
    )
    request = old_llm.create_request(messages, config)
    assert request.tool_choice == ClaudeToolChoice(type="tool", name="parse_data")
    assert request.system is not None
    assert len(request.system) == 1

    llm = BedrockClaudeLLM(model_id=CLAUDE_5_5_OPUS, client=make_test_bedrock_runtime_client([]))
    request = llm.create_request(messages, config)
    assert request.tool_choice == ClaudeToolChoice(type="auto")
    assert request.system is not None
    assert [c.text for c in request.system if isinstance(c, ClaudeTextContent)] == [
        "Be helpful.",
        "You must call the `parse_data` tool in your response.",
    ]
