import base64
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field
from rich import print as rich_print

from e84_geoai_common.llm import models
from e84_geoai_common.llm.core.llm import (
    Base64ImageContent,
    CachePointContent,
    ExecutableLLMTool,
    JSONContent,
    LLMAssistantMessage,
    LLMInferenceConfig,
    LLMTool,
    LLMToolChoice,
    LLMToolResultContent,
    LLMToolUseContent,
    LLMUserMessage,
    RedactedThinkingContent,
    TextContent,
    ThinkingContent,
)
from e84_geoai_common.llm.models import claude
from e84_geoai_common.llm.models.claude import (
    CLAUDE_3_5_SONNET_V2,
    CLAUDE_4_5_HAIKU,
    CLAUDE_4_7_OPUS,
    CLAUDE_5_5_HAIKU,
    CLAUDE_5_5_OPUS,
    BedrockClaudeLLM,
    ClaudeInvokeLLMRequest,
    ClaudeRedactedThinkingContent,
    ClaudeTextContent,
    ClaudeThinkingContent,
    ClaudeToolChoice,
    get_claude_model_capabilities,
)
from e84_geoai_common.llm.models.converse import BedrockConverseLLM
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


def test_thinking_blocks_are_kept_and_sent_back() -> None:
    # Always uses canned responses. Real models only return thinking blocks when thinking is
    # enabled, which this library doesn't do.
    llm = BedrockClaudeLLM(
        model_id=CLAUDE_5_5_HAIKU,
        client=_MockBedrockRuntimeClient(
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
    assert resp.content == [
        ThinkingContent(thinking="", signature="abc123"),
        RedactedThinkingContent(data="encrypted"),
        TextContent(text="olleh"),
    ]
    assert resp.to_text_only() == "olleh"

    # The thinking blocks are sent back when the assistant message is part of the conversation.
    request = llm.create_request(
        [LLMUserMessage(content="hello"), resp, LLMUserMessage(content="again")],
        LLMInferenceConfig(),
    )
    assert request.messages[1].content[:2] == [
        ClaudeThinkingContent(type="thinking", thinking="", signature="abc123"),
        ClaudeRedactedThinkingContent(type="redacted_thinking", data="encrypted"),
    ]


def test_model_ids_are_normalized() -> None:
    expected = get_claude_model_capabilities(CLAUDE_4_7_OPUS)
    assert expected is not None
    for model_id in [
        "global.anthropic.claude-opus-4-7",
        "eu.anthropic.claude-opus-4-7",
        "anthropic.claude-opus-4-7",
        "arn:aws:bedrock:us-east-1:123456789012:inference-profile/global.anthropic.claude-opus-4-7",
        "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-opus-4-7",
    ]:
        assert get_claude_model_capabilities(model_id) == expected, model_id
    assert get_claude_model_capabilities("meta.llama3-70b-instruct-v1:0") is None


def test_every_model_constant_has_capabilities() -> None:
    constants = {
        name: value
        for name, value in vars(claude).items()
        if name.startswith("CLAUDE_") and isinstance(value, str)
    }
    assert [name for name, v in constants.items() if not get_claude_model_capabilities(v)] == []
    # They're also exported from the models package.
    assert [name for name in constants if not hasattr(models, name)] == []


def test_caching_uses_normalized_model_id() -> None:
    model_id = CLAUDE_3_5_SONNET_V2.replace("us.", "global.", 1)
    llm = BedrockClaudeLLM(model_id=model_id, client=make_test_bedrock_runtime_client([]))
    request = llm.create_request(
        [LLMUserMessage(content="hello")], LLMInferenceConfig(system_prompt="Be helpful.")
    )
    assert request.system == [ClaudeTextContent(text="Be helpful.")]


def test_sampling_params_omitted_for_newer_models() -> None:
    messages = [LLMUserMessage(content="hello")]
    old_llm = BedrockClaudeLLM(
        model_id=CLAUDE_4_5_HAIKU, client=make_test_bedrock_runtime_client([])
    )
    assert old_llm.create_request(messages, LLMInferenceConfig()).temperature == 0

    for model_id in [CLAUDE_5_5_OPUS, "global.anthropic.claude-opus-5-5"]:
        llm = BedrockClaudeLLM(model_id=model_id, client=make_test_bedrock_runtime_client([]))
        assert llm.create_request(messages, LLMInferenceConfig()).temperature is None
        # An explicit default temperature is dropped too.
        assert llm.create_request(messages, LLMInferenceConfig(temperature=0)).temperature is None
        with pytest.raises(ValueError, match="doesn't support"):
            llm.create_request(messages, LLMInferenceConfig(temperature=0.5))
        with pytest.raises(ValueError, match="doesn't support"):
            llm.create_request(messages, LLMInferenceConfig(top_p=0.5))


def test_converse_omits_sampling_params_for_newer_claude_models() -> None:
    llm = BedrockConverseLLM(model_id=CLAUDE_5_5_OPUS, client=make_test_bedrock_runtime_client([]))
    request = llm.create_request([LLMUserMessage(content="hello")], LLMInferenceConfig())
    assert request.inferenceConfig is not None
    assert request.inferenceConfig.temperature is None


def test_request_temperature_defaults_to_none() -> None:
    assert ClaudeInvokeLLMRequest().temperature is None


class Country(BaseModel):
    name: str


class City(BaseModel):
    name: str
    country: Country
    previous_country: Country | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class CityWithDefault(BaseModel):
    name: str
    tags: list[str] = Field(default=["type", "object"], examples=[{"type": "object"}])


def _structured_output_schema(model: type[BaseModel]) -> dict[str, Any]:
    llm = BedrockClaudeLLM(client=make_test_bedrock_runtime_client([]))
    request = llm.create_request(
        [LLMUserMessage(content="hello")], LLMInferenceConfig(structured_output_model=model)
    )
    assert request.output_config is not None
    return request.output_config.format.schema_


def test_structured_output_schema_disallows_additional_properties() -> None:
    schema = _structured_output_schema(Country)
    assert schema["additionalProperties"] is False

    # Data keywords like default and examples are left alone.
    schema = _structured_output_schema(CityWithDefault)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["tags"]["default"] == ["type", "object"]
    assert schema["properties"]["tags"]["examples"] == [{"type": "object"}]

    # Objects with arbitrary keys can't be represented.
    with pytest.raises(ValueError, match="additionalProperties"):
        _structured_output_schema(City)


def test_disallow_additional_properties_handles_nested_and_nullable_objects() -> None:
    schema = claude._disallow_additional_properties(  # type: ignore[reportPrivateUsage]
        {
            "type": "object",
            "properties": {
                "a": {"type": ["object", "null"], "properties": {}},
                "b": {"anyOf": [{"$ref": "#/$defs/C"}, {"type": "null"}]},
            },
            "$defs": {"C": {"type": "object", "default": {"type": "object"}}},
        }
    )
    assert schema == {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "a": {"type": ["object", "null"], "properties": {}, "additionalProperties": False},
            "b": {"anyOf": [{"$ref": "#/$defs/C"}, {"type": "null"}]},
        },
        "$defs": {
            "C": {"type": "object", "default": {"type": "object"}, "additionalProperties": False}
        },
    }


def _parse_city_config(tool_choice: LLMToolChoice) -> LLMInferenceConfig:
    return LLMInferenceConfig(
        system_prompt="Be helpful.",
        tools=[
            LLMTool(
                name="parse_data",
                description="Parse the data.",
                input_model=Country,
                output_model=None,
            )
        ],
        tool_choice=tool_choice,
    )


def test_forced_tool_choice_requires_a_configured_tool() -> None:
    llm = BedrockClaudeLLM(client=make_test_bedrock_runtime_client([]))
    for tool_name in [None, "other_tool"]:
        config = _parse_city_config(
            LLMToolChoice(mode="force_specific_tool_use", tool_name=tool_name)
        )
        with pytest.raises(ValueError, match="not one of the configured tools"):
            llm.create_request([LLMUserMessage(content="hello")], config)


def test_forced_tool_choice_replaced_for_models_that_reject_it() -> None:
    messages = [LLMUserMessage(content="hello")]
    config = _parse_city_config(
        LLMToolChoice(mode="force_specific_tool_use", tool_name="parse_data")
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


def test_unforced_tool_choice_retries_until_tool_use() -> None:
    tool_use = {"type": "tool_use", "id": "t1", "name": "parse_data", "input": {"name": "France"}}
    client = _MockBedrockRuntimeClient(
        [
            claude_response_with_content("France is a country."),
            claude_response_with_content([tool_use], overrides={"stop_reason": "tool_use"}),
        ]
    )
    llm = BedrockClaudeLLM(model_id=CLAUDE_5_5_OPUS, client=client)
    config = _parse_city_config(LLMToolChoice(mode="force_tool_use", tool_name=None))
    resp = llm.prompt([LLMUserMessage(content="France")], config)

    assert resp.content == [LLMToolUseContent(id="t1", name="parse_data", input={"name": "France"})]
    assert client.canned_responses == []
    # Token counts cover both attempts.
    assert resp.metadata is not None
    assert (resp.metadata.input_tokens, resp.metadata.output_tokens) == (246, 642)


def test_unforced_tool_choice_gives_up_after_retries() -> None:
    client = _MockBedrockRuntimeClient([claude_response_with_content("No.") for _ in range(3)])
    llm = BedrockClaudeLLM(model_id=CLAUDE_5_5_OPUS, client=client)
    config = _parse_city_config(LLMToolChoice(mode="force_tool_use", tool_name=None))
    resp = llm.prompt([LLMUserMessage(content="France")], config)
    assert isinstance(resp, LLMAssistantMessage)
    assert resp.to_text_only() == "No."
    assert client.canned_responses == []
