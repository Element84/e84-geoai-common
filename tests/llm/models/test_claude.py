import base64
import json
from pathlib import Path

from pydantic import BaseModel, Field
from rich import print as rich_print

from e84_geoai_common.llm.core.llm import (
    Base64ImageContent,
    JSONContent,
    LLMInferenceConfig,
    LLMMessage,
    LLMTool,
    LLMToolResultContent,
    LLMToolUseContent,
    TextContent,
)
from e84_geoai_common.llm.models.claude import (
    BedrockClaudeLLM,
)
from e84_geoai_common.llm.tests.mock_bedrock import (
    _MockBedrockRuntimeClient,  # type: ignore[reportPrivateUsage]
    claude_response_with_content,
    make_test_bedrock_client,
)


def test_basic_usage() -> None:
    llm = BedrockClaudeLLM(client=make_test_bedrock_client([claude_response_with_content("olleh")]))
    config = LLMInferenceConfig()
    resp = llm.prompt(
        [LLMMessage(content="Output the word hello backwards and only that.")], config
    )
    expected_resp = LLMMessage(role="assistant", content=[TextContent(text="olleh")])
    assert resp == expected_resp


def test_with_response_prefix() -> None:
    llm = BedrockClaudeLLM(client=make_test_bedrock_client([claude_response_with_content("  15")]))
    config = LLMInferenceConfig(response_prefix="5 + 10 =")
    resp = llm.prompt(
        [LLMMessage(content="Output the sum of 5 and 10 without additional explanation")], config
    )
    assert resp == LLMMessage(role="assistant", content=[TextContent(text="5 + 10 =  15")])


def test_json_mode() -> None:
    json_mode_prompt = """
        Create a list of the numbers 1 through 5.

        Here's an example of the desired output for the number 2 through 6
        {"result": [2, 3, 4, 5, 6]}
    """
    llm = BedrockClaudeLLM(
        client=make_test_bedrock_client(
            [claude_response_with_content('"result": [1, 2, 3, 4, 5]}')]
        )
    )
    config = LLMInferenceConfig(json_mode=True)
    resp = llm.prompt([LLMMessage(content=json_mode_prompt)], config)

    assert resp.role == "assistant"
    assert len(resp.content) == 1
    content = resp.content[0]
    assert isinstance(content, TextContent)
    assert json.loads(content.text) == {"result": [1, 2, 3, 4, 5]}


def encode_image_to_base64_str(image_path: str | Path) -> str:
    image_path = Path(image_path)
    with image_path.open("rb") as image_file:
        encoded_bytes = base64.b64encode(image_file.read())
    return encoded_bytes.decode("utf-8")


def test_image_input() -> None:
    llm = BedrockClaudeLLM(client=make_test_bedrock_client([claude_response_with_content("cat")]))

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

    tool = LLMTool(
        name="GetWeatherInfo",
        description="Get current weather info for a place.",
        input_model=GetWeatherInfoInput,
        output_model=WeatherInfo,
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
            overrides={"stop_reason": "tool_use"},
        ),
    ]
    client = make_test_bedrock_client(dummy_responses)
    llm = BedrockClaudeLLM(client=client)
    config = LLMInferenceConfig(tools=[tool])

    # test tool use
    messages = [LLMMessage(content=prompt)]
    resp = llm.prompt(messages=messages, inference_cfg=config)

    # print out solution if doing a live test, so we can inspect it if the test fails
    if not isinstance(client, _MockBedrockRuntimeClient):
        rich_print(resp)

    assert isinstance(resp.content, list)
    tool_use_req = resp.content[-1]
    assert isinstance(tool_use_req, LLMToolUseContent)
    assert tool_use_req.name == tool.name
    tool_inputs = GetWeatherInfoInput.model_validate(tool_use_req.input)
    assert "philadelphia" in tool_inputs.place.lower()

    # test tool result
    tool_result = WeatherInfo(weather_description="Sunny", temperature=20, humidity=50)
    tool_result_content = LLMToolResultContent(
        id=tool_use_req.id, content=[JSONContent(data=tool_result.model_dump())]
    )
    messages = [*messages, resp, LLMMessage(content=[tool_result_content])]
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

    tool = LLMTool(
        name="GenerateImage",
        description="Generate an image from text. Returns the generated image only.",
        input_model=ImageGeneratorInput,
        output_model=None,
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
            overrides={"stop_reason": "tool_use"},
        ),
    ]
    client = make_test_bedrock_client(dummy_responses)
    llm = BedrockClaudeLLM(client=client)
    config = LLMInferenceConfig(tools=[tool])

    # test tool use
    messages = [LLMMessage(content=prompt)]
    resp = llm.prompt(messages=messages, inference_cfg=config)

    # print out solution if doing a live test, so we can inspect it if the test fails
    if not isinstance(client, _MockBedrockRuntimeClient):
        rich_print(resp)

    assert isinstance(resp.content, list)
    tool_use_req = resp.content[-1]
    assert isinstance(tool_use_req, LLMToolUseContent)
    assert tool_use_req.name == tool.name
    _ = ImageGeneratorInput.model_validate(tool_use_req.input)

    # test tool result
    image_path = str(Path(__file__).parent / "images/cat.webp")
    base64_string = encode_image_to_base64_str(image_path)
    image_content = Base64ImageContent(media_type="image/webp", data=base64_string)

    tool_result_content = LLMToolResultContent(id=tool_use_req.id, content=[image_content])
    messages = [*messages, resp, LLMMessage(content=[tool_result_content])]
    resp = llm.prompt(messages=messages, inference_cfg=config)

    # print out solution if doing a live test, so we can inspect it if the test fails
    if not isinstance(client, _MockBedrockRuntimeClient):
        rich_print(resp)

    assert "cat" in resp.to_text_only().lower()
