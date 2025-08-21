import base64
import json
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from textwrap import dedent

from pydantic import BaseModel, Field
from rich import print as rich_print

from e84_geoai_common.llm.core.llm import (
    Base64ImageContent,
    CachePointContent,
    ExecutableLLMTool,
    JSONContent,
    LLMInferenceConfig,
    LLMMessage,
    LLMTool,
    LLMToolResultContent,
    LLMToolUseContent,
    TextContent,
)
from e84_geoai_common.llm.models.claude import (
    CLAUDE_BEDROCK_MODEL_IDS,
    BedrockClaudeLLM,
    ClaudeTextContent,
)
from e84_geoai_common.llm.tests.mock_bedrock_runtime import (
    _MockBedrockRuntimeClient,  # type: ignore[reportPrivateUsage]
    claude_response_with_content,
    make_test_bedrock_runtime_client,
)


def test_basic_usage() -> None:
    llm = BedrockClaudeLLM(
        client=make_test_bedrock_runtime_client([claude_response_with_content("olleh")])
    )
    config = LLMInferenceConfig()
    resp = llm.prompt(
        [LLMMessage(content="Output the word hello backwards and only that.")], config
    )
    expected_resp = LLMMessage(role="assistant", content=[TextContent(text="olleh")])
    assert resp == expected_resp


def test_with_response_prefix() -> None:
    llm = BedrockClaudeLLM(
        client=make_test_bedrock_runtime_client([claude_response_with_content("  15")])
    )
    config = LLMInferenceConfig(response_prefix="5 + 10 =")
    resp = llm.prompt(
        [LLMMessage(content="Output the sum of 5 and 10 without additional explanation")], config
    )
    assert resp == LLMMessage(role="assistant", content=[TextContent(text="5 + 10 =  15")])


def test_json_mode() -> None:
    json_mode_prompt = dedent("""
        Create a list of the numbers 1 through 5.

        Here's an example of the desired output for the number 2 through 6
        {"result": [2, 3, 4, 5, 6]}
    """)
    llm = BedrockClaudeLLM(
        client=make_test_bedrock_runtime_client(
            [claude_response_with_content('"result": [1, 2, 3, 4, 5]}\n')]
        )
    )
    config = LLMInferenceConfig(json_mode=True)
    resp = llm.prompt([LLMMessage(content=json_mode_prompt)], config)

    assert resp.role == "assistant"
    assert len(resp.content) == 1
    content = resp.content[0]
    assert isinstance(content, TextContent)
    assert json.loads(content.text) == {"result": [1, 2, 3, 4, 5]}


def test_json_mode_no_extra_text() -> None:
    prompt = dedent("""
        Generate some fake weather data as a JSON and then write a brief
        weather report based on it.

        Example JSON output:
        {
            "temperature_degC": 7,
            "humidity_pct": 25,
            "air_quality_index": 50
        }
    """)
    stub_response = dedent("""
        "temperature_degC": 7,
        "humidity_pct": 25,
        "air_quality_index": 50
    }""")
    llm = BedrockClaudeLLM(
        client=make_test_bedrock_runtime_client([claude_response_with_content(stub_response)])
    )
    config = LLMInferenceConfig(json_mode=True)
    resp = llm.prompt([LLMMessage(content=prompt)], config)

    assert resp.role == "assistant"
    with does_not_raise():
        _ = json.loads(resp.to_text_only())


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

    prompt_message = LLMMessage(
        role="user",
        content=[prompt_text, image_content],
    )

    config = LLMInferenceConfig()

    resp = llm.prompt([prompt_message], config)

    assert resp == LLMMessage(role="assistant", content=[TextContent(text="cat")])


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

    def exec_weather_tool(
        context: None, tool_use_request: LLMToolUseContent
    ) -> LLMToolResultContent:
        weather_info = WeatherInfo(weather_description="Sunny", temperature=20, humidity=50)
        tool_result = LLMToolResultContent(
            id=tool_use_request.id, content=[JSONContent(data=weather_info.model_dump())]
        )
        return tool_result

    tool = ExecutableLLMTool(
        tool_def=LLMTool(
            name="GetWeatherInfo",
            description="Get current weather info for a place.",
            input_model=GetWeatherInfoInput,
            output_model=WeatherInfo,
        ),
        execution_func=exec_weather_tool,
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
    config = LLMInferenceConfig(tools=[tool.tool_def])

    # test tool use
    messages = [LLMMessage(content=prompt)]
    resp = llm.prompt(messages=messages, inference_cfg=config)

    # print out solution if doing a live test, so we can inspect it if the test fails
    if not isinstance(client, _MockBedrockRuntimeClient):
        rich_print(resp)

    assert isinstance(resp.content, list)
    tool_use_req = resp.content[-1]
    assert isinstance(tool_use_req, LLMToolUseContent)
    assert tool_use_req.name == tool.tool_def.name
    tool_inputs = GetWeatherInfoInput.model_validate(tool_use_req.input)
    assert "philadelphia" in tool_inputs.place.lower()

    # test tool result
    tool_result = tool.execute(None, tool_use_req)
    messages = [*messages, resp, LLMMessage(content=[tool_result])]
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

    def exec_image_gen_tool(
        context: None, tool_use_request: LLMToolUseContent
    ) -> LLMToolResultContent:
        image_path = str(Path(__file__).parent / "images/cat.webp")
        base64_string = encode_image_to_base64_str(image_path)
        image_content = Base64ImageContent(media_type="image/webp", data=base64_string)
        tool_result = LLMToolResultContent(id=tool_use_request.id, content=[image_content])
        return tool_result

    tool = ExecutableLLMTool(
        tool_def=LLMTool(
            name="GenerateImage",
            description="Generate an image from text. Returns the generated image only.",
            input_model=ImageGeneratorInput,
            output_model=None,
        ),
        execution_func=exec_image_gen_tool,
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
    config = LLMInferenceConfig(tools=[tool.tool_def])

    # test tool use
    messages = [LLMMessage(content=prompt)]
    resp = llm.prompt(messages=messages, inference_cfg=config)

    # print out solution if doing a live test, so we can inspect it if the test fails
    if not isinstance(client, _MockBedrockRuntimeClient):
        rich_print(resp)

    assert isinstance(resp.content, list)
    tool_use_req = resp.content[-1]
    assert isinstance(tool_use_req, LLMToolUseContent)
    assert tool_use_req.name == tool.tool_def.name
    _ = ImageGeneratorInput.model_validate(tool_use_req.input)

    # test tool result
    tool_result = tool.execute(None, tool_use_req)
    messages = [*messages, resp, LLMMessage(content=[tool_result])]
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
    resp = llm.prompt([LLMMessage(content=[text_content, CachePointContent()])], config)
    expected_resp = LLMMessage(role="assistant", content=[TextContent(text="olleh")])
    assert resp == expected_resp


def test_large_system_prompt() -> None:
    """This test is most interesting as a live-test.

    It validates that the cache control headers indicate that prompt caching actually worked.

    It uses a large system prompt so that the minimum token limit is reached.
    """
    long_system_prompt_path = Path(__file__).parent / "long_system_prompt.txt"
    with long_system_prompt_path.open(encoding="utf-8") as file:
        system_prompt = file.read()

        text_content = TextContent(text="Output the word hello backwards and only that.")
        llm = BedrockClaudeLLM(
            model_id=CLAUDE_BEDROCK_MODEL_IDS["Claude 3.5 Haiku"],
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
        config = LLMInferenceConfig(system_prompt=system_prompt)
        request = llm.create_request(
            [LLMMessage(content=[text_content, CachePointContent()])], config
        )
        response = llm.invoke_model_with_request(request)

        assert response.content == [ClaudeTextContent(text="olleh")]
        assert response.usage.cache_creation_input_tokens is not None
        assert response.usage.cache_creation_input_tokens > 0
