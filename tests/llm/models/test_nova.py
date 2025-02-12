from e84_geoai_common.llm.core.llm import LLMInferenceConfig, LLMMessage
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
    assert resp == LLMMessage(role="assistant", content="olleh")


def test_with_response_prefix() -> None:
    llm = BedrockNovaLLM(client=make_test_bedrock_client([nova_response_with_content("15")]))
    config = LLMInferenceConfig(response_prefix="5 + 10 = ")
    resp = llm.prompt(
        [LLMMessage(content="Output the sum of 5 and 10 without additional explanation")], config
    )
    assert resp == LLMMessage(role="assistant", content="5 + 10 = 15")


def test_json_mode() -> None:
    json_mode_prompt = """
        Create a list of the numbers 1 through 5.
    """
    llm = BedrockNovaLLM(
        client=make_test_bedrock_client([nova_response_with_content("[1, 2, 3, 4, 5]\n```")])
    )
    config = LLMInferenceConfig(json_mode=True)
    resp = llm.prompt([LLMMessage(content=json_mode_prompt)], config)
    assert resp == LLMMessage(role="assistant", content="[1, 2, 3, 4, 5]\n")
