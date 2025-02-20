import json
from pathlib import Path

from e84_geoai_common.llm.core.llm import (
    Base64ImageContent,
    LLMInferenceConfig,
    LLMMessage,
    TextContent,
)
from e84_geoai_common.llm.models.converse import BedrockConverseLLM
from e84_geoai_common.llm.tests.mock_bedrock import (
    converse_response_with_content,
    make_test_bedrock_client,
)


def test_basic_usage() -> None:
    llm = BedrockConverseLLM(client=make_test_bedrock_client([
        converse_response_with_content("olleh")
    ]))
    config = LLMInferenceConfig()
    resp = llm.prompt(
        [LLMMessage(content="Output the word hello backwards and only that.")], config
    )
    assert resp == LLMMessage(role="assistant", content=[TextContent(text="olleh")])



def test_with_response_prefix() -> None:
    llm = BedrockConverseLLM(client=make_test_bedrock_client([
        converse_response_with_content("  15")
    ]))
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
    llm = BedrockConverseLLM(
        client=make_test_bedrock_client([
            converse_response_with_content('"result": [1, 2, 3, 4, 5]}'
        )])
    )
    config = LLMInferenceConfig(json_mode=True)
    resp = llm.prompt([LLMMessage(content=json_mode_prompt)], config)
    assert resp.role == "assistant"
    assert len(resp.content) == 1
    content = resp.content[0]
    assert isinstance(content, TextContent)
    assert json.loads(content.text) == {"result": [1, 2, 3, 4, 5]}

#Unlike other SDK APIs, Converse API doesn't support base64 encoding of images
def encode_image_to_bytes(image_path: str) -> bytes:
    image = Path(image_path)
    with image.open("rb") as image_file:
        base64_encoded = image_file.read()
    return base64_encoded

def test_image_input() -> None:
    # Simulated response when processing an image
    expected_text_output = "cat"
    mock_response = converse_response_with_content(expected_text_output)

    llm = BedrockConverseLLM(client=make_test_bedrock_client([mock_response]))

    #locally ai generated picture of a cat
    image_path = str(Path(__file__).parent / "images/cat.webp")
    base64_string = encode_image_to_bytes(image_path)

    # Mock image content
    image_content = Base64ImageContent(media_type="image/webp", data=base64_string)
    prompt_text = TextContent(
        text = """
        Report the animal in the picture and only that. I.e. dog.
        Respond in lowercase and with only one word, the animal in the picture.
    """
    )

    prompt_message = LLMMessage(
        role="user",
        content=[prompt_text, image_content],
    )

    config = LLMInferenceConfig()

    resp = llm.prompt([prompt_message], config)

    assert resp == LLMMessage(role="assistant", content=[TextContent(text="cat")])

#tool used in testing tool usage
def output__string(first_word: str, second_word: str) -> str:
    """Outputs a string, given two words."""
    return first_word + " " + second_word + " element eightyfour"

def test_tool_usage() -> None:
    llm = BedrockConverseLLM(
        client=make_test_bedrock_client([
            converse_response_with_content("result: element 84 element eightyfour")
        ])
    )

    prompt_message = LLMMessage(
        role="user",
        content="""use your tool and respond with: "result: output"
        Input two words, "element" and "84".""",
    )

    config = LLMInferenceConfig(
        system_prompt=(
            "You can use tools. "
        ),
        tools=[output__string],
    )

    resp = llm.prompt([prompt_message], config)

    expected_resp = LLMMessage(
        role="assistant",
        content=[TextContent(text="result: element 84 element eightyfour")],
    )

    assert resp == expected_resp
