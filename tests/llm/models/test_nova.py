import json

from e84_geoai_common.llm.core.llm import LLMInferenceConfig, LLMMessage, TextContent
from e84_geoai_common.llm.models.nova import BedrockNovaLLM
from e84_geoai_common.llm.tests.mock_bedrock import (
    make_test_bedrock_client,
    nova_response_with_content,
)


def test_basic_usage() -> None:
    llm = BedrockNovaLLM(client=make_test_bedrock_client([nova_response_with_content("olleh")]))
    config = LLMInferenceConfig()
    resp = llm.prompt(
        [LLMMessage(content="Output the word hello backwards and only that.")], config
    )
    expected_resp = LLMMessage(role="assistant", content=[TextContent(text="olleh")])
    assert resp == expected_resp


def test_json_mode() -> None:
    json_mode_prompt = """
        Create a list of the numbers 1 through 5.
    """
    llm = BedrockNovaLLM(
        client=make_test_bedrock_client([nova_response_with_content("[1, 2, 3, 4, 5]\n```")])
    )
    config = LLMInferenceConfig(json_mode=True)
    resp = llm.prompt([LLMMessage(content=json_mode_prompt)], config)

    assert resp.role == "assistant"
    assert len(resp.content) == 1
    content = resp.content[0]
    assert isinstance(content, TextContent)
    assert json.loads(content.text) == [1, 2, 3, 4, 5]
