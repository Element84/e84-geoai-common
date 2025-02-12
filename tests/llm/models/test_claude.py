import json

from e84_geoai_common.llm.core.llm import LLMInferenceConfig, LLMMessage, TextContent
from e84_geoai_common.llm.models.claude import (
    BedrockClaudeLLM,
)
from e84_geoai_common.llm.tests.mock_bedrock import (
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
