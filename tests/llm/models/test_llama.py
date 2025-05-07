from e84_geoai_common.llm.core.llm import (
    LLMInferenceConfig,
    LLMMessage,
)
from e84_geoai_common.llm.models.llama import BedrockLlamaLLM
from e84_geoai_common.llm.tests.mock_bedrock import (
    llama_response_with_content,
    make_test_bedrock_client,
)


def test_basic_usage() -> None:
    llm = BedrockLlamaLLM(client=make_test_bedrock_client([llama_response_with_content("olleh")]))
    config = LLMInferenceConfig()
    resp = llm.prompt(
        [LLMMessage(content="Output the word hello backwards and only that.")], config
    )
    assert resp == LLMMessage(role="assistant", content="olleh")


def test_with_response_prefix() -> None:
    llm = BedrockLlamaLLM(client=make_test_bedrock_client([llama_response_with_content("15")]))
    config = LLMInferenceConfig(response_prefix="5 + 10 = ")
    resp = llm.prompt(
        [LLMMessage(content="Output the sum of 5 and 10 without additional explanation")], config
    )
    assert resp == LLMMessage(role="assistant", content="5 + 10 = 15")
