import base64
import json
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from textwrap import dedent

from e84_geoai_common.llm.core.llm import (
    Base64ImageContent,
    CachePointContent,
    LLMInferenceConfig,
    LLMUserMessage,
    TextContent,
)
from e84_geoai_common.llm.models.nova import (
    BedrockNovaLLM,
    NovaMessage,
    NovaResponseOutput,
    NovaTextContent,
)
from e84_geoai_common.llm.tests.mock_bedrock_runtime import (
    make_test_bedrock_runtime_client,
    nova_response_with_content,
)


def test_basic_usage() -> None:
    llm = BedrockNovaLLM(
        client=make_test_bedrock_runtime_client([nova_response_with_content("olleh")])
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


def test_with_response_prefix() -> None:
    llm = BedrockNovaLLM(
        client=make_test_bedrock_runtime_client([nova_response_with_content("15")])
    )
    config = LLMInferenceConfig(response_prefix="5 + 10 = ")
    resp = llm.prompt(
        [LLMUserMessage(content="Output the sum of 5 and 10 without additional explanation")],
        config,
    )
    assert resp.model_dump(exclude={"metadata": {"input_tokens", "output_tokens"}}) == {
        "role": "assistant",
        "content": [{"text": "5 + 10 = 15"}],
        "metadata": {"stop_reason": "end_turn"},
    }


def test_json_mode() -> None:
    json_mode_prompt = dedent("""
        Create a list of the numbers 1 through 5.

        Here's an example of the desired output for the number 2 through 6
        {"result": [2, 3, 4, 5, 6]}
    """)
    llm = BedrockNovaLLM(
        client=make_test_bedrock_runtime_client(
            [nova_response_with_content('"result": [1, 2, 3, 4, 5]}\n')]
        )
    )
    config = LLMInferenceConfig(json_mode=True)
    resp = llm.prompt([LLMUserMessage(content=json_mode_prompt)], config)

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
    llm = BedrockNovaLLM(
        client=make_test_bedrock_runtime_client([nova_response_with_content(stub_response)])
    )
    config = LLMInferenceConfig(json_mode=True)
    resp = llm.prompt([LLMUserMessage(content=prompt)], config)

    assert resp.role == "assistant"
    with does_not_raise():
        _ = json.loads(resp.to_text_only())


def encode_image_to_base64_str(image_path: str) -> str:
    image = Path(image_path)
    with image.open("rb") as image_file:
        encoded_bytes = base64.b64encode(image_file.read())
    return encoded_bytes.decode("utf-8")


def test_image_input() -> None:
    llm = BedrockNovaLLM(
        client=make_test_bedrock_runtime_client([nova_response_with_content("cat")])
    )

    # locally ai generated picture of a cat
    image_path = str(Path(__file__).parent / "images/cat.webp")
    base64_string = encode_image_to_base64_str(image_path)

    image_content = Base64ImageContent(media_type="image/webp", data=base64_string)
    prompt_text = TextContent(
        text="""
        Report the animal in the picture and only that, in lowercase.
        I.e. dog. Respond with ONLY one word, the animal in the picture.
    """
    )

    prompt_message = LLMUserMessage(content=[prompt_text, image_content])

    config = LLMInferenceConfig()

    resp = llm.prompt([prompt_message], config)

    assert resp.model_dump(exclude={"metadata": {"input_tokens", "output_tokens"}}) == {
        "role": "assistant",
        "content": [{"text": "cat"}],
        "metadata": {"stop_reason": "end_turn"},
    }


def test_basic_usage_with_prompt_caching() -> None:
    text_content = TextContent(text="Output the word hello backwards and only that.")
    llm = BedrockNovaLLM(
        client=make_test_bedrock_runtime_client([nova_response_with_content("olleh")])
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
    long_system_prompt_path = Path(__file__).parent / "long_system_prompt.txt"
    with long_system_prompt_path.open(encoding="utf-8") as file:
        system_prompt = file.read()

        text_content = TextContent(text="Output the word hello backwards and only that.")
        llm = BedrockNovaLLM(
            client=make_test_bedrock_runtime_client([nova_response_with_content("olleh")]),
        )
        config = LLMInferenceConfig(system_prompt=system_prompt)
        request = llm.create_request(
            [LLMUserMessage(content=[text_content, CachePointContent()])], config
        )
        response = llm.invoke_model_with_request(request)

        assert response.output == NovaResponseOutput(
            message=NovaMessage(
                role="assistant",
                content=[NovaTextContent(text="olleh", cache_point=None)],
            )
        )

        assert response.usage.cache_write_input_token_count is not None
        assert response.usage.cache_write_input_token_count > 0
